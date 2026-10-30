"""Health of the encoder states: global and temporal collapse"""

import torch.nn.functional as F

from h2r.metrics.stats import Moments


def straightness(emb):
    """Temporal straightness (LeWM eq. 9): mean cosine similarity between consecutive
    latent velocities v_t = z_t+1 - z_t; 1 = straight path
    """
    v = (emb[:, 1:] - emb[:, :-1]).float()  # (B, T-1, D)
    return F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1).mean()


class RepresentationHealth:
    """
    - sigreg: the regularizer on the validation states;
    - rankme: effective rank of the states;
    - alpha: decay of their spectrum;
    - dead_dims: dimensions whose variance is below 0.01 (SIGReg pulls it to 1);
    - temporal_change: E|s_t+1 - s_t|^2 / (2 Var(s)), 1 for independent frames, 0 when nothing moves within a
      clip. SIGReg compares clips of the batch, so it does not stop the encoder from erasing all motion, which
      makes the prediction trivial;
    - straightness.
    """

    def __init__(self):
        self.states = Moments()
        self.n, self.sigreg, self.change, self.straight = 0, 0.0, 0.0, 0.0

    def update(self, emb, sigreg):
        """emb: (B, T, D)"""
        b = len(emb)
        self.states.update(emb.flatten(0, 1))
        self.sigreg += sigreg.float() * b
        self.change += (emb[:, 1:] - emb[:, :-1]).float().pow(2).mean() * b
        self.straight += straightness(emb) * b
        self.n += b

    def compute(self):
        var = self.states.var()
        return {
            "sigreg": float(self.sigreg / self.n),
            "rankme": self.states.rankme(),
            "alpha": self.states.alpha(),
            "dead_dims": int((var < 0.01).sum()),
            "temporal_change": float(self.change / self.n) / (2 * float(var.mean())),
            "straightness": float(self.straight / self.n),
        }
