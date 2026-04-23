import os
import numpy as np
import torch
from .load_model import fetch_sss, fetch_tap
from .model.tokenizer import ABtokenizer
from .model.flag_calibrator import FlagCalibrator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CALIBRATOR_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'flag_calibrators.pkl')


class SSSResult:
    """Result from FlashABB-SSS: per-residue structure-aware embeddings."""

    def __init__(self, embeddings: torch.Tensor, mask: torch.Tensor):
        self._embeddings = embeddings
        self._mask = mask

    @property
    def embeddings(self) -> torch.Tensor:
        """(batch, seq_len, emb_size) per-residue embeddings."""
        return self._embeddings

    @property
    def mask(self) -> torch.Tensor:
        """(batch, seq_len) bool mask — True where residue is present."""
        return self._mask


class TAPResult:
    """Result from FlashTAP: four antibody developability scores and flag probabilities."""

    TAP_COLS = ['PSH', 'PPC', 'PNC', 'SFvCSP']

    def __init__(self, tensor: torch.Tensor, flag_probs_array: np.ndarray | None = None):
        self._tensor = tensor
        self._flag_probs_array = flag_probs_array  # (batch, 4) or None

    @property
    def tensor(self) -> torch.Tensor:
        """(batch, 4) raw score tensor."""
        return self._tensor

    @property
    def scores(self) -> list[dict]:
        """List of dicts (one per antibody) mapping property name → float."""
        return [
            {col: self._tensor[i, j].item() for j, col in enumerate(self.TAP_COLS)}
            for i in range(self._tensor.shape[0])
        ]

    @property
    def flag_probs(self) -> list[dict] | None:
        """List of dicts (one per antibody) mapping property name → P(flag).

        Returns None if no calibrator was loaded.
        """
        if self._flag_probs_array is None:
            return None
        return [
            {col: float(self._flag_probs_array[i, j]) for j, col in enumerate(self.TAP_COLS)}
            for i in range(self._flag_probs_array.shape[0])
        ]

    @property
    def any_flag_prob(self) -> list[float] | None:
        """P(any flag) for each antibody, assuming property independence.

        Returns None if no calibrator was loaded.
        """
        if self._flag_probs_array is None:
            return None
        any_flag = 1 - np.prod(1 - self._flag_probs_array, axis=1)
        return any_flag.tolist()


def _tokenize(seqs, alphabet: ABtokenizer, device):
    tokens = alphabet(seqs, pad=True, w_extra_tkns=False)
    return tokens.to(device)


def _emb_and_mask(model, seqs, tokens, alphabet, device):
    """Run BERTCoords forward and return (embeddings, mask) with sep removed."""
    pad_mask = tokens.eq(alphabet.pad_token).to(device)
    emb = model(seqs, tokens, pad_mask, return_emb=True)

    sep_mask = tokens != alphabet.sep_token
    src_shape = list(tokens.shape)
    src_shape[1] -= 1
    mask = (~pad_mask)[sep_mask].view(src_shape)
    return emb, mask


class pretrained_sss:
    """FlashABB-SSS: structure-aware antibody sequence encoder.

    Usage::

        from flash_abb import pretrained_sss
        sss = pretrained_sss()
        result = sss(['EVQL...|DIQL...'])
        print(result.embeddings.shape)   # (1, seq_len, 128)
    """

    def __init__(self, random_init: bool = False, device=DEVICE):
        self.device = device
        self.sss = fetch_sss(random_init=random_init, device=str(device))
        self.sss.eval()
        self.sss.requires_grad_(False)
        self.alphabet = self.sss.alphabet

    def __call__(self, seqs, batch_size: int = 50) -> SSSResult:
        all_emb, all_mask = [], []
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            tokens = _tokenize(batch, self.alphabet, self.device)
            with torch.no_grad():
                emb, mask = _emb_and_mask(self.sss, batch, tokens, self.alphabet, self.device)
            all_emb.append(emb)
            all_mask.append(mask)
        return SSSResult(torch.cat(all_emb), torch.cat(all_mask))


class pretrained_tap:
    """FlashTAP: predicts four TAP developability scores from antibody sequences.

    Scores: PSH (patches of surface hydrophobicity), PPC (positive patches),
    PNC (negative patches), SFvCSP (structural Fv charge symmetry parameter).

    Usage::

        from flash_abb import pretrained_tap
        tap = pretrained_tap()
        result = tap(['EVQL...|DIQL...'])
        print(result.scores)        # [{'PSH': ..., 'PPC': ..., 'PNC': ..., 'SFvCSP': ...}]
        print(result.flag_probs)    # [{'PSH': 0.12, 'PPC': 0.03, 'PNC': 0.05, 'SFvCSP': 0.41}]
        print(result.any_flag_prob) # [0.52]
    """

    def __init__(self, random_init: bool = False, device=DEVICE):
        self.device = device
        self.encoder, self.head = fetch_tap(random_init=random_init, device=str(device))
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.head.eval()
        self.head.requires_grad_(False)
        self.alphabet = self.encoder.alphabet
        self.calibrator = FlagCalibrator.load(_CALIBRATOR_PATH) if not random_init else None

    def __call__(self, seqs, batch_size: int = 50) -> TAPResult:
        all_scores = []
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            tokens = _tokenize(batch, self.alphabet, self.device)
            with torch.no_grad():
                emb, mask = _emb_and_mask(self.encoder, batch, tokens, self.alphabet, self.device)
                scores = self.head(emb, mask)
            all_scores.append(scores)
        score_tensor = torch.cat(all_scores)

        flag_probs_array = None
        if self.calibrator is not None:
            scores_np = score_tensor.cpu().numpy()
            flag_probs_array = self.calibrator.predict_proba(scores_np)

        return TAPResult(score_tensor, flag_probs_array)
