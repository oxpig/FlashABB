# Copyright 2024 Exscientia
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import sys
from functools import reduce
from operator import mul
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from ..openfold.model.heads import PerResidueLDDTCaPredictor
from ..openfold.model.primitives import (
    LayerNorm,
    Linear,
    ipa_point_weights_init_,
)
from ..openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from ..openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from ..openfold.utils.precision_utils import is_fp16_enabled
from ..openfold.utils.rigid_utils import Rigid, Rotation
from ..openfold.utils.tensor_utils import (
    dict_multimap,
    flatten_final_dims,
    permute_final_dims,
)

from .flashpoint_attention import FlashpointAttention

# torch._dynamo.config.capture_scalar_outputs = True


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        # self.linear_1 = Linear(self.c, 2 * self.c, init="relu")
        # self.linear_2 = Linear(2 * self.c, 2 * self.c, init="relu")
        # self.linear_3 = Linear(2 * self.c, self.c, init="final")
        self.linear_1 = Linear(self.c, 4 * self.c, init="relu")
        self.linear_3 = Linear(4 * self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        # s = self.linear_2(s)
        # s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        embed_dim,
        padding_idx,
        c_ipa,
        no_heads_ipa,
        no_blocks,
        no_qk_points=4,
        no_v_points=8,
        dropout_rate=0.1,
        no_transition_layers=1,
        epsilon=1.0e-07 ,
        inf=10000000.0 ,
        rotation_propagation=True,
        use_original_sm=True,
        use_plddt=False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            embed_dim:
                Initial embedding dimension
            c_ipa:
                IPA hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            rotation_propagation:
                If true allow rigid gradients to propogate
            use_original_sm:
                If True use original structure module implementation else use ABB3. If True:
                    Use bias in attention
                    Correctly implement line 11 of algorithm 20
                    Number of linear layers in AngleResnetBlock is 2 instead of 3
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.epsilon = epsilon
        self.inf = inf
        self.rotation_propagation = rotation_propagation
        self.use_original_sm = use_original_sm
        self.use_plddt = use_plddt

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        # remove to match ABB3 and because inputs are one hot encodings.
        # self.layer_norm_s = LayerNorm(self.c_s)
        # self.layer_norm_s = LayerNorm(self.embed_dim)
        # self.layer_norm_z = LayerNorm(self.c_z)

        # self.linear_in_node = Linear(self.c_s, self.embed_dim)
        self.embed_tokens = nn.Embedding(
            self.c_s,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.ipa_layers = nn.ModuleList(
            [
                FlashpointAttention(
                    self.embed_dim,
                    self.c_ipa,
                    self.no_heads_ipa,
                    self.no_qk_points,
                    self.no_v_points,
                    inf=self.inf,
                    eps=self.epsilon,
                    ipa_bias=self.use_original_sm,
                    # use_spectra=False,
                )
                for _ in range(self.no_blocks)
            ]
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa_layers = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(self.no_blocks)]
        )

        self.transition_layers = nn.ModuleList(
            [
                StructureModuleTransition(
                    self.embed_dim,
                    self.no_transition_layers,
                    self.dropout_rate,
                )
                for _ in range(self.no_blocks)
            ]
        )

        # self.bb_update_layers = nn.ModuleList(
        #     [BackboneUpdate(self.embed_dim) for _ in range(self.no_blocks)]
        # )

        if self.use_plddt:
            self.plddt = PerResidueLDDTCaPredictor(
                no_bins=50, c_in=self.embed_dim, c_hidden=256
            )

    def forward(
        self,
        src,
        init_src=None,
        coords=None,
        aatype=None,
        res_idx=None,
        # mask=None,
        inplace_safe=False,
        _offload_inference=False,
        return_attn_weights=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        # s = evoformer_output_dict["single"]

        # if mask is None:
        #     # [*, N]
        #     mask = src.new_ones(src.shape[:-1])
        mask = (~src.eq(self.padding_idx)).long()
        # mask = src.eq(self.padding_idx)

        # Removed to make closer to ABB3 and because the inputs are one hot encodings.
        # [*, N, C_s]

        # [*, N, embed_dim]
        s = self.embed_tokens(src)
        if init_src is not None:
            s = s + init_src
        # s = s.transpose(0,1)
        # s = self.layer_norm_s(s)
        # s_initial = s

        # [*, N]
        if coords is None:
            rigids = Rigid.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                self.training,
                fmt="quat",
            )
        else:
            X_n = coords[...,0,:]
            X_ca = coords[...,1,:]
            X_c = coords[...,2,:]
            rigids = Rigid.from_3_points(X_n, X_ca, X_c)
        outputs = []
        attn_weights = []
        for i in range(self.no_blocks):
            s_diff, layer_weights = self.ipa_layers[i](
                s,
                # res_idx,
                None,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                return_attn_weights=return_attn_weights,
            )
            s = s + s_diff
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa_layers[i](s)
            s = self.transition_layers[i](s)

            # [*, N]
            # line 10 of algorithm 20
            # rigids = rigids.compose_q_update_vec(self.bb_update_layers[i](s))

            preds = {
                "states": s,
            }
            attn_weights.append(layer_weights)

            outputs.append(preds)

            # if not self.rotation_propagation:
            #     rigids = rigids.stop_rot_gradient()

        # return s.transpose(0,1)
        show_3d = False
        # show_3d = True
        if show_3d:
            from matplotlib import pyplot as plt
            colour = True
            aas = [int(tok) for tok in src[0]]
            # print(aas)
            # eos_idx = aas.index(2)
            eos_idx = -1
            # colour = False
            # ca = rigids.get_trans()[0,...].detach().cpu().numpy()
            ca = rigids.get_trans()[0,...]
            dist = (ca[:,None,:] - ca[None,:,:]).norm(dim=-1)
            # contact = (dist < 8).int().detach().cpu().numpy()
            # contact = (dist < 6).int().detach().cpu().numpy()
            contact = (8-dist).clamp(min=0).int().detach().cpu().numpy()
            # contact = (12-dist).clamp(min=0).int().detach().cpu().numpy()
            plt.imshow(contact[1:eos_idx, 1:eos_idx])
            plt.show()
            ca = ca.detach().cpu().numpy()
            # raise OSError
            # ca = rigids.get_trans()[3,...].detach().cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_box_aspect((1, 1, 1))
            ax.scatter(
                ca[1:eos_idx,0],
                ca[1:eos_idx,1],
                ca[1:eos_idx,2],
                marker='o'
            )
            ax.plot(
                ca[1:eos_idx,0],
                ca[1:eos_idx,1],
                ca[1:eos_idx,2],
                color='blue'
            )
            # split_idx = aas.index(30)
            # # Heavy chain
            # ax.scatter(
            #     ca[1:split_idx,0],
            #     ca[1:split_idx,1],
            #     ca[1:split_idx,2],
            #     marker='o'
            # )
            # ax.plot(
            #     ca[1:split_idx,0],
            #     ca[1:split_idx,1],
            #     ca[1:split_idx,2],
            #     color='blue'
            # )
            # # Light chain
            # ax.scatter(
            #     ca[split_idx+1:eos_idx,0],
            #     ca[split_idx+1:eos_idx,1],
            #     ca[split_idx+1:eos_idx,2],
            #     marker='^'
            # )
            # ax.plot(
            #     ca[split_idx+1:eos_idx,0],
            #     ca[split_idx+1:eos_idx,1],
            #     ca[split_idx+1:eos_idx,2],
            #     color='orange'
            # )
            # "Linker"
            # ax.plot(
            #     ca[split_idx-1:split_idx+2,0],
            #     ca[split_idx-1:split_idx+2,1],
            #     ca[split_idx-1:split_idx+2,2],
            #     color='green'
            # )
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()
            # raise OSError
        return s, rigids, attn_weights
        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        if self.use_plddt:
            outputs["plddt"] = self.plddt(s)
        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
