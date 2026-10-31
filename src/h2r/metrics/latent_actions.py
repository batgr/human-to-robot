"""Latent actions of the IDM: richness, use by the predictor, leakage of the future, transfer"""

import torch

from h2r.metrics.stats import Moments


def other(x):
    """The same tensor for another clip of the batch (the validation order is shuffled)"""
    return x.roll(1, dims=0)


def mse(a, b):
    return (a.float() - b.float()).pow(2).mean()


class LatentActions:
    """
    - kl: KL of the latent actions to the N(0, 1) prior;
    - active_dims: dimensions of mu whose variance is above 0.01 (Burda et al. 2016);
    - rankme: effective rank of mu;
    - controllability: 1 - err(z) / err(z of another clip), 0 when the predictor ignores z;
    - scene_cut: error on the last transition when its frame comes from another clip, over the error on the real
      one (Garrido et al. 2026, section 6). A z that carries the whole next frame keeps it near 1; their models
      reach x2.0-2.5;
    - cycle: z of clip A applied to clip B, re-inferred from B's predictions, applied back to A: error over the
      error with A's own z (Garrido et al. 2026, section 6; x1.03-1.34 for them), 1 = actions that transfer.
    """

    def __init__(self):
        self.mu = Moments()
        self.n, self.kl = 0, 0.0
        self.err = self.err_other = self.err_last = self.err_cut = self.err_cycle = 0.0

    def update(self, model, emb, z, pred, kl):
        """emb: (B, T, D) or (B, T, N, D), z: (B, T-1, z_dim) mu of the IDM, pred = predict(emb[:, :-1], z)"""
        b, target = len(emb), emb[:, 1:]
        self.mu.update(z.flatten(0, 1))
        self.kl += kl.float() * b
        self.err += mse(pred, target) * b
        self.err_other += mse(model.predict(emb[:, :-1], other(z)), target) * b

        cut = emb.clone()
        cut[:, -1] = other(emb)[:, -1]  # scene change on the last transition
        z_cut, _ = model.idm(cut)
        self.err_cut += mse(model.predict(cut[:, :-1], z_cut)[:, -1], cut[:, -1]) * b
        self.err_last += mse(pred[:, -1], target[:, -1]) * b

        clip_b = other(emb)
        pred_b = model.predict(clip_b[:, :-1], z)  # A's actions on B
        pairs = torch.stack([clip_b[:, :-1], pred_b], dim=2).flatten(
            0, 1
        )  # (B * (T-1), 2, D) or (B * (T-1), 2, N, D)
        z_back, _ = model.idm(pairs)  # (B * (T-1), 1, z_dim)
        pred_a = model.predict(emb[:, :-1], z_back.view_as(z))
        self.err_cycle += mse(pred_a, target) * b
        self.n += b

    def compute(self):
        return {
            "kl": float(self.kl / self.n),
            "active_dims": int((self.mu.var() > 0.01).sum()),
            "rankme": self.mu.rankme(),
            "controllability": float(1 - self.err / self.err_other),
            "scene_cut": float(self.err_cut / self.err_last),
            "cycle": float(self.err_cycle / self.err),
        }
