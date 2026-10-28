"""Health of the encoder states: global, spatial and temporal collapse"""

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
    - rankme_token, rankme_frame: effective rank of the tokens and of the frames (mean of their tokens);
    - alpha: decay of the token spectrum;
    - dead_dims: dimensions whose variance is below 0.01 (SIGReg pulls it to 1);
    - token_cosine: spatial collapse, 1 when the tokens of a frame are identical;
    - temporal_change: E|s_t+1 - s_t|^2 / (2 Var(s)), 1 for independent frames, 0 when nothing moves within a
      clip. SIGReg compares clips of the batch, so it does not stop the encoder from erasing all motion, which
      makes the prediction trivial;
    - straightness.
    """

    def __init__(self):
        self.tokens, self.frames = Moments(), Moments()
        self.n, self.sigreg, self.cosine, self.change, self.straight = 0, 0.0, 0.0, 0.0, 0.0

    def update(self, emb, sigreg):
        """emb: (B, T, N, D)"""
        b = len(emb)
        self.tokens.update(emb.flatten(0, 2))
        self.frames.update(emb.mean(2).flatten(0, 1))
        self.sigreg += sigreg.float() * b
        self.cosine += token_cosine(emb) * b
        self.change += (emb[:, 1:] - emb[:, :-1]).float().pow(2).mean() * b
        self.straight += straightness(emb) * b
        self.n += b

    def compute(self):
        var = self.tokens.var()
        return {
            "sigreg": float(self.sigreg / self.n),
            "rankme_token": self.tokens.rankme(),
            "rankme_frame": self.frames.rankme(),
            "alpha": self.tokens.alpha(),
            "dead_dims": int((var < 0.01).sum()),
            "token_cosine": float(self.cosine / self.n),
            "temporal_change": float(self.change / self.n) / (2 * float(var.mean())),
            "straightness": float(self.straight / self.n),
        }
