import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention, _score_mod_signature
except ImportError:
    pass  # flex_attention requires PyTorch >= 2.5; unused in the inference path

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from .openfold.model.primitives import (
    LayerNorm,
    Linear,
    ipa_point_weights_init_,
)
from .openfold.utils.rigid_utils import Rigid, Rotation
from .openfold.utils.tensor_utils import (
    dict_multimap,
    flatten_final_dims,
    permute_final_dims,
)


class FlashpointAttention(nn.Module):
    """
    Implements Algorithm TODO.
    """

    def __init__(
        self,
        # max_pos_diff: int,
        c_s: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        ipa_bias: bool = True,
        use_spectra=True,
    ):
        """
        Args:
            max_pos_diff:
                Maximum positional encoding difference
            c_s:
                Single representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
            ipa_bias:
                Use bias in linear layers.
        """
        super(FlashpointAttention, self).__init__()

        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.ipa_bias = ipa_bias

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=self.ipa_bias)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=self.ipa_bias)

        h_pts = self.no_heads * (2*self.no_qk_points + self.no_v_points) * 3
        self.linear_points = Linear(self.c_s, h_pts, bias=self.ipa_bias)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3

        hpv = self.no_heads * self.no_v_points * 3

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        self.rel_pos_dim = 128 if use_spectra else 0
        self.rotary_emb = RotaryEmbedding(dim=c_hidden)
        self.use_spectra = use_spectra
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_hidden + self.no_v_points * 4 + self.rel_pos_dim
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        # dists,
        res_idx,
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        q = rearrange(q, 'b l n d -> b n l d')
        k = rearrange(k, 'b l n d -> b n l d')

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        pts = self.linear_points(s)
        pts = torch.split(pts, pts.shape[-1] // 3, dim=-1)
        pts = torch.stack(pts, dim=-1)

        # [*, N_res, H, (2 * P_q + P_v), 3]
        pts = pts.view(pts.shape[:-2] + (self.no_heads, -1, 3))

        q_pts, k_pts, kv_pts = torch.split(
            pts, [
                self.no_qk_points,
                self.no_qk_points,
                self.no_v_points,
            ], dim=-2
        )

        q_pts = r[..., None, None].apply(q_pts)
        k_pts = r[..., None, None].apply(k_pts)
        kv_pts = r[..., None, None].apply(kv_pts)

        q_pts_enc = flatten_final_dims(torch.cat((
            q_pts**2,
            -2*q_pts,
            torch.ones_like(q_pts)
        ), dim=-1), 2)
        k_pts_enc = flatten_final_dims(torch.cat((
            torch.ones_like(k_pts),
            k_pts,
            k_pts**2
        ), dim=-1), 2)
        v_pts_enc = rearrange(kv_pts, 'b l h v_pts d -> b l h (v_pts d)')
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(q_pts_enc.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        q_pts_enc = q_pts_enc * head_weights * (-0.5)
        q_pts_enc = rearrange(q_pts_enc, 'b l n d -> b n l d')
        k_pts_enc = rearrange(k_pts_enc, 'b l n d -> b n l d')
        q *= math.sqrt(1.0 / (3 * self.c_hidden))
        q = torch.cat((q, q_pts_enc), dim=-1)
        k = torch.cat((k, k_pts_enc), dim=-1)
        v = torch.cat((v, v_pts_enc), dim=-1)

        ##########################
        # Compute attention scores
        ##########################
        # square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        # square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]

        v = rearrange(v, 'b l n d -> b n l d')
        if self.use_spectra:
            kv_idx = res_idx % self.rel_pos_dim
            kv_idx = torch.nn.functional.one_hot(kv_idx, self.rel_pos_dim)
            v = torch.cat((v, kv_idx[:, None, :, :].expand(v.shape[0], v.shape[1], -1, -1)), dim=-1)
        attn_mask = self.inf * (mask-1)
        attn_mask = attn_mask[:,None,None,:].to(q.dtype)
        # MPS scaled_dot_product_attention requires d_q == d_v; here d_q=52, d_v=168.
        # Fall back to manual matmul+softmax whenever dimensions differ.
        if q.shape[-1] != v.shape[-1]:
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) + attn_mask
            o = torch.matmul(attn_weights.softmax(dim=-1), v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        o = rearrange(o, 'b n l d -> b l n d')

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        if self.use_spectra:
            o, o_pt, o_pos = torch.split(o, [self.c_hidden, 3*8, self.rel_pos_dim], dim=-1) # TODO: params
        else:
            o, o_pt = torch.split(o, [self.c_hidden, 3*8], dim=-1) # TODO: params
        o_pt = rearrange(o_pt, 'b l h (v_pts d) -> b l h v_pts d', d=3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if self.use_spectra:
            # Shift position spectra so it gives relative position
            pos_idxs = torch.arange(o_pos.shape[-1], device=q.device)[None,:].expand(res_idx.shape[0], -1)
            q_idx = (res_idx[:, :, None] + pos_idxs[:, None, :]) % self.rel_pos_dim
            q_idx = q_idx[:, :, None, :].expand(o_pos.shape)
            o_pos = torch.gather(o_pos, -1, q_idx)
            o_pos = flatten_final_dims(o_pos, 2)

            # [*, N_res, C_s]
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pos), dim=-1).to(
                    dtype=o.dtype
                )
            )
        else:
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                    dtype=o.dtype
                )
            )

        return s
