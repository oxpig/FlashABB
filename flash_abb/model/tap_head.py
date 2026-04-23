import torch
import torch.nn as nn


class TAPHead(nn.Module):
    """Per-residue MLP regression head for TAP property prediction.

    Predicts four developability scores (PSH, PPC, PNC, SFvCSP) via masked
    sum pooling over per-residue predictions.  Normalisation statistics are
    stored as buffers so they travel with the model weights.
    """

    TAP_COLS = ['PSH', 'PPC', 'PNC', 'SFvCSP']

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )
        self.register_buffer('tgt_mean', torch.zeros(4))
        self.register_buffer('tgt_std', torch.ones(4))

    def forward(self, per_residue_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_residue_emb: (batch, seq_len, input_dim)
            mask: (batch, seq_len) — True where residue is present
        Returns:
            (batch, 4) denormalised TAP scores
        """
        per_res = self.mlp(per_residue_emb)
        summed = (per_res * mask.unsqueeze(-1).float()).sum(dim=1)
        return summed * self.tgt_std + self.tgt_mean
