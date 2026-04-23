import torch
import torch.nn as nn
from torch import Tensor

from .tokenizer import ABtokenizer
from .fpa_transformer.internal_structure_transformer import StructureModule


class BERTCoords(nn.Module):
    """Structure-aware antibody sequence encoder (FlashABB-SSS).

    Frozen FlashABB predicts backbone coordinates; an IPA transformer then
    produces per-residue embeddings that capture both sequence and geometry.
    """

    def __init__(
        self,
        num_heads: int = 12,
        num_layers: int = 6,
        emb_size: int = 128,
        dropout: float = 0.0,
        use_coords: bool = True,
        device: str = 'cpu',
    ):
        super().__init__()
        self.emb_size = emb_size
        self.use_coords = use_coords

        # Imported lazily to avoid circular import (pretrained → BERTCoords → pretrained)
        from flash_abb.pretrained import pretrained as FlashABBPretrained
        self.folder = FlashABBPretrained(device=device)
        self.folder.flabb.requires_grad_(False)

        self.alphabet = ABtokenizer()
        self.encoder = StructureModule(
            c_s=len(self.alphabet.aa_to_token),
            embed_dim=emb_size,
            padding_idx=self.alphabet.pad_token,
            c_ipa=16,
            no_heads_ipa=num_heads,
            no_blocks=num_layers,
            dropout_rate=dropout,
        )
        self.norm = nn.LayerNorm(emb_size)
        self.generator = nn.Linear(emb_size, len(self.alphabet.aa_to_token))

    def forward(
        self,
        src_seq,
        src: Tensor,
        mask=None,
        return_emb: bool = False,
        return_attn_weights: bool = False,
    ):
        sep_mask = src != self.alphabet.sep_token
        src_shape = list(src.shape)
        src_shape[1] = src.shape[1] - 1

        if self.use_coords:
            coords = self.folder(src_seq).bb_coords / 10

        src_emb, rigids, attn_weights = self.encoder(
            src[sep_mask].view(src_shape),
            coords=coords if self.use_coords else None,
            return_attn_weights=return_attn_weights,
        )

        if return_emb:
            return self.norm(src_emb)

        logits = self.generator(self.norm(src_emb))
        if return_attn_weights:
            return logits, attn_weights
        return logits, rigids
