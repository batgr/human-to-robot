"""Health of the encoder states: global and temporal collapse"""

import torch.nn.functional as F

from h2r.metrics.stats import Moments


def straightness(emb):
    """Temporal straightness (LeWM eq. 9): mean cosine similarity between consecutive
    latent velocities v_t = z_t+1 - z_t, a frame being all its tokens; 1 = straight path
    """
    v = (emb[:, 1:] - emb[:, :-1]).flatten(2).float()  # (B, T-1, N*D)
    return F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1).mean()


def token_cosine(emb):
    """Mean cosine between two different tokens of the same frame: 1 when the N tokens are copies of one vector"""
    n = emb.size(2)
    u = F.normalize(emb.float(), dim=-1).sum(2)  # (B, T, D)
    return ((u.pow(2).sum(-1) - n) / (n * (n - 1))).mean()


class RepresentationHealth:
    """
    - sigreg: the regularizer on the validation states;
    - rankme: effective rank of the states (of the tokens with the patch projector);
    - alpha: decay of their spectrum;
    - dead_dims: dimensions whose variance is below 0.01 (SIGReg pulls it to 1);
    - temporal_change: E|s_t+1 - s_t|^2 / (2 Var(s)), 1 for independent frames, 0 when nothing moves within a
      clip. SIGReg compares clips of the batch, so it does not stop the encoder from erasing all motion, which
      makes the prediction trivial;
    - straightness;
    - token_cosine (patch projector only): spatial collapse, 1 when the tokens of a frame are identical.
    """

    def __init__(self):
        self.states = Moments()
        self.n, self.sigreg, self.change, self.straight, self.cosine = 0, 0.0, 0.0, 0.0, None

    def update(self, emb, sigreg):
        """emb: (B, T, D) or (B, T, N, D)"""
        b = len(emb)
        self.states.update(emb.flatten(0, -2))
        if emb.dim() == 4:
            self.cosine = (self.cosine or 0.0) + token_cosine(emb) * b
        self.sigreg += sigreg.float() * b
        self.change += (emb[:, 1:] - emb[:, :-1]).float().pow(2).mean() * b
        self.straight += straightness(emb) * b
        self.n += b

    def compute(self):
        var = self.states.var()
        tokens = {} if self.cosine is None else {"token_cosine": float(self.cosine / self.n)}
        return {
            "sigreg": float(self.sigreg / self.n),
            "rankme": self.states.rankme(),
            "alpha": self.states.alpha(),
            "dead_dims": int((var < 0.01).sum()),
            "temporal_change": float(self.change / self.n) / (2 * float(var.mean())),
            "straightness": float(self.straight / self.n),
        } | tokens
