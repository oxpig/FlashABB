import pickle
import numpy as np

TAP_COLS = ['PSH', 'PPC', 'PNC', 'SFvCSP']


class FlagCalibrator:
    """Inference-only flag probability calibrator backed by isotonic regression."""

    def __init__(self, calibrators, psh_safe_center=None):
        self.calibrators = calibrators
        self.psh_safe_center = psh_safe_center

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            calibrators=data['calibrators'],
            psh_safe_center=data.get('psh_safe_center'),
        )

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Map (n, 4) TAP score array to (n, 4) flag probability array."""
        probs = np.zeros_like(scores, dtype=np.float32)
        for i, col in enumerate(TAP_COLS):
            vals = scores[:, i]
            if col == 'PSH':
                p_low = self.calibrators[col]['low'].predict(vals)
                p_high = self.calibrators[col]['high'].predict(vals)
                probs[:, i] = 1 - (1 - p_low) * (1 - p_high)
            else:
                probs[:, i] = self.calibrators[col].predict(vals)
        return probs

    def predict_any_flag(self, scores: np.ndarray) -> np.ndarray:
        """Return (n,) array of P(any flag) assuming property independence."""
        probs = self.predict_proba(scores)
        return 1 - np.prod(1 - probs, axis=1)
