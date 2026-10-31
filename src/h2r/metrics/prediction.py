"""Prediction quality, one step with the true past (teacher forcing), against the trivial prediction"""

from h2r.metrics.stats import Moments


class PredictionQuality:
    """
    - tf: MSE of the next state with the IDM's latent actions, over every transition of the clip;
    - persistence: MSE of the trivial prediction s_t+1 = s_t on the same transitions, the level to beat;
    - std_ratio: std of the predictions / std of the targets, near 0 when the predictor outputs the mean.
    """

    def __init__(self):
        self.preds, self.targets = Moments(), Moments()
        self.n, self.err, self.persistence = 0, 0.0, 0.0

    def update(self, emb, pred):
        """emb: (B, T, D) or (B, T, N, D) states, pred: predictions of emb[:, 1:]"""
        b = len(emb)
        target = emb[:, 1:].float()
        self.err += (pred.float() - target).pow(2).mean() * b
        self.persistence += (emb[:, :-1].float() - target).pow(2).mean() * b
        self.preds.update(pred.flatten(0, -2))
        self.targets.update(target.flatten(0, -2))
        self.n += b

    def compute(self):
        ratio = (self.preds.var().sum() / self.targets.var().sum()).sqrt()
        return {
            "tf": float(self.err / self.n),
            "persistence": float(self.persistence / self.n),
            "std_ratio": float(ratio),
        }
