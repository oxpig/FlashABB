import numpy as np
import torch

from .load_model import load_model
from .model import dlc
from .model.flash_abb import featurize, FlashABBResult


def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class pretrained:

    def __init__(self, model_to_use="flash-abb", random_init=False, device=None, dlc_correction=True):
        device = device or _default_device()
        super().__init__()

        self.used_device = torch.device(device)

        self.flabb, self.hparams = load_model(model_to_use, random_init=random_init)
        self.flabb.to(self.used_device)
        self.flabb.eval() # Default
        self.device = torch.device(device)

        # Dropout-LayerNorm Correction (DLC): fixes a small systematic
        # eval-mode bias in StructureModule's dropout->LayerNorm sites.
        # See flash_abb/model/dlc.py and
        # https://arxiv.org/abs/2609.32062. On by default; hooks are
        # always installed but gated by this flag so it can be toggled
        # at runtime via the dlc_correction property.
        self._dlc_mode_flag = {"value": "on" if dlc_correction else "off"}
        dlc.install(self.flabb.model, mode_flag=self._dlc_mode_flag)

    @property
    def dlc_correction(self):
        return self._dlc_mode_flag["value"] == "on"

    @dlc_correction.setter
    def dlc_correction(self, value):
        self._dlc_mode_flag["value"] = "on" if value else "off"

    def freeze(self):
        self.flabb.eval()

    def unfreeze(self):
        self.flabb.train()

    def from_features(self, features, batch_size=50):
        pred = self.flabb.model(
            {'single': features['single']},
            features['aatype'],
            features['res_idx'],
            features['mask']
        )
        result = FlashABBResult(seqs, pred, features['mask'])
        return result

    def __call__(self, seqs, batch_size=50):
        features = featurize(seqs, self.device)
        pred = self.flabb.model(
            {'single': features['single']},
            features['aatype'],
            features['res_idx'],
            features['mask']
        )
        result = FlashABBResult(seqs, pred, features['mask'])
        return result
