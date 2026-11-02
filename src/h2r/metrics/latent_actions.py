"""Latent actions of the IDM: richness, use by the predictor, leakage of the future, transfer"""

import torch
import torch.nn.functional as F

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
    - mu_spread_over_sigma: std of mu over the transitions / the IDM's mean sigma, per dimension, averaged: below 1 the
      sampled z of training is mostly noise;
    - controllability: 1 - err(z) / err(z of another clip), 0 when the predictor ignores z;
    - leak: z of clip A applied to clip B, share of the transitions whose prediction lands closer to A's next state
      than to B's: z carries A's next frame, not an action (diagnostics E8: 1 % with teacher forcing, 66 % with the
      rollout loss K = 2). About 0.5 at initialization, when the prediction is near neither: read its fall, then any
      rise;
    - transfer_cos: median cosine between the change A's z causes on B (pred - s_t of B) and A's true change, on the
      third of A's transitions that move most in the states: the motion of A, transferred (E3: 0.31 at beta 1e-4);
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
        self.sigma = 0.0
        self.raw = {k: [] for k in ["leak", "transfer_cos", "motion_a"]}

    def update(self, model, emb, z, pred, kl):
        """emb: (B, T, D) or (B, T, N, D), z: (B, T-1, z_dim) mu of the IDM, pred = predict(emb[:, :-1], z)"""
        b, target = len(emb), emb[:, 1:]
        self.mu.update(z.flatten(0, 1))
        self.kl += kl.float() * b
        self.err += mse(pred, target) * b
        pred_ab = model.predict(emb[:, :-1], other(z))  # A = the other clip, its z on B
        self.err_other += mse(pred_ab, target) * b
        e, a, p = emb.float().flatten(2), other(emb).float().flatten(2), pred_ab.float().flatten(2)
        new = {
            "leak": (p - a[:, 1:]).norm(dim=-1) < (p - e[:, 1:]).norm(dim=-1),
            "transfer_cos": F.cosine_similarity(p - e[:, :-1], a[:, 1:] - a[:, :-1], dim=-1),
            "motion_a": 1 - F.cosine_similarity(a[:, :-1], a[:, 1:], dim=-1),
        }
        for k, v in new.items():
            self.raw[k].append(v.float().flatten().cpu())
        _, logvar = model.idm.posterior(emb)
        self.sigma += (0.5 * logvar.float()).exp().flatten(0, 1).sum(0)

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
        r = {k: torch.cat(v) for k, v in self.raw.items()}
        moving = r["motion_a"] >= r["motion_a"].quantile(2 / 3)
        sigma = self.sigma.cpu().double() / self.mu.n
        return {
            "kl": float(self.kl / self.n),
            "mu_spread_over_sigma": float((self.mu.var().clamp_min(0).sqrt() / sigma).mean()),
            "leak": float(r["leak"].mean()),
            "transfer_cos": float(r["transfer_cos"][moving].median()),
            "active_dims": int((self.mu.var() > 0.01).sum()),
            "rankme": self.mu.rankme(),
            "controllability": float(1 - self.err / self.err_other),
            "scene_cut": float(self.err_cut / self.err_last),
            "cycle": float(self.err_cycle / self.err),
        }
