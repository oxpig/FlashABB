import torch
from .load_model import fetch_sss, fetch_tap
from .model.tokenizer import ABtokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """Result from FlashTAP: four antibody developability scores."""

    TAP_COLS = ['PSH', 'PPC', 'PNC', 'SFvCSP']

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

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
        print(result.scores)   # [{'PSH': ..., 'PPC': ..., 'PNC': ..., 'SFvCSP': ...}]
    """

    def __init__(self, random_init: bool = False, device=DEVICE):
        self.device = device
        self.encoder, self.head = fetch_tap(random_init=random_init, device=str(device))
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.head.eval()
        self.head.requires_grad_(False)
        self.alphabet = self.encoder.alphabet

    def __call__(self, seqs, batch_size: int = 50) -> TAPResult:
        all_scores = []
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            tokens = _tokenize(batch, self.alphabet, self.device)
            with torch.no_grad():
                emb, mask = _emb_and_mask(self.encoder, batch, tokens, self.alphabet, self.device)
                scores = self.head(emb, mask)
            all_scores.append(scores)
        return TAPResult(torch.cat(all_scores))
