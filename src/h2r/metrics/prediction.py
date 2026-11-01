"""Prediction quality: one step with the true past (teacher forcing) and open-loop unrolling, against the trivial
prediction"""

import torch

from h2r.metrics.stats import Moments


class PredictionQuality:
    """
    - tf: MSE of the next state with the IDM's latent actions, over every transition of the clip;
    - persistence: MSE of the trivial prediction s_t+1 = s_t on the same transitions, the level to beat;
    - std_ratio: std of the predictions / std of the targets, near 0 when the predictor outputs the mean;
    - tf_static, persistence_static, tf_moving, persistence_moving: the same on the third of the transitions with the
      least / the most pixel change (mean absolute change between the two frames, cut at the thirds of the whole
      validation set: the same split for every run and every encoder). The static half of the data hides the moving
      one in the means (diagnostics E1-E3: tf / persistence 2.6 static, 0.33 moving);
    - change_ratio_static, change_ratio_moving: median of |pred - s_t| / |s_t+1 - s_t|, above 1 = invented motion
      (1.8 on static clips, diagnostics E2-E5), below 1 = shrunk motion (0.5 when the KL starves z, E3);
    - retrieval_moving: share of the moving transitions whose prediction is closer to s_t+1 than to every other frame
      of the clip (chance 1 / T; E5-E6).
    """

    def __init__(self):
        self.preds, self.targets = Moments(), Moments()
        self.n, self.err, self.persistence = 0, 0.0, 0.0
        self.raw = {k: [] for k in ["motion", "err", "persistence", "change_ratio", "retrieval"]}

    def update(self, emb, pred, clip):
        """emb: (B, T, D) or (B, T, N, D) states, pred: predictions of emb[:, 1:], clip: (B, T, 3, H, W) pixels"""
        b = len(emb)
        target = emb[:, 1:].float()
        self.err += (pred.float() - target).pow(2).mean() * b
        self.persistence += (emb[:, :-1].float() - target).pow(2).mean() * b
        self.preds.update(pred.flatten(0, -2))
        self.targets.update(target.flatten(0, -2))
        self.n += b

        e, p = emb.float().flatten(2), pred.float().flatten(2)  # one vector per frame, its tokens concatenated
        d_true, d_pred = e[:, 1:] - e[:, :-1], p - e[:, :-1]
        frames = torch.arange(1, e.size(1), device=e.device)
        new = {
            "motion": (clip[:, 1:].float() - clip[:, :-1].float()).abs().mean((2, 3, 4)),
            "err": (p - e[:, 1:]).pow(2).mean(-1),
            "persistence": d_true.pow(2).mean(-1),
            "change_ratio": d_pred.norm(dim=-1) / d_true.norm(dim=-1).clamp_min(1e-8),
            "retrieval": (torch.cdist(p, e).argmin(-1) == frames).float(),
        }
        for k, v in new.items():
            self.raw[k].append(v.flatten().cpu())

    def compute(self):
        ratio = (self.preds.var().sum() / self.targets.var().sum()).sqrt()
        r = {k: torch.cat(v) for k, v in self.raw.items()}
        low, high = r["motion"].quantile(torch.tensor([1 / 3, 2 / 3]))
        thirds = {"static": r["motion"] <= low, "moving": r["motion"] >= high}
        out = {
            "tf": float(self.err / self.n),
            "persistence": float(self.persistence / self.n),
            "tf_over_persistence": float(self.err / self.persistence),
            "std_ratio": float(ratio),
        }
        for name, keep in thirds.items():
            err, pers = r["err"][keep].mean(), r["persistence"][keep].mean()
            out |= {
                f"tf_{name}": float(err),
                f"persistence_{name}": float(pers),
                f"tf_over_persistence_{name}": float(err / pers),
                f"change_ratio_{name}": float(r["change_ratio"][keep].median()),
            }
        return out | {"retrieval_moving": float(r["retrieval"][thirds["moving"]].mean())}


class RolloutError:
    """Open-loop unrolling from one real state, JEPA-WM's test protocol (Terver et al. 2025): the predictor reads back
    its own predictions in a window of history_size states (W_p), with the IDM's latent actions (mu) of the true transitions.
    - error_h{h}: MSE of the state h steps ahead, h = 1..T-1;
    - persistence_h{h}: MSE of copying the first state (s_0 for s_h), the level to beat;
    - skill_h{h} = 1 - error_h{h} / persistence_h{h}: 0 = as good as copying, 1 = perfect;
    - growth_h{h} = error_h{h} / error_h{h-1}: growth of the error per step (the effective Lipschitz constant of the
      unrolled predictor, their section D), which the rollout loss should lower.
    """

    def __init__(self):
        self.n, self.err, self.persistence = 0, 0.0, 0.0

    def update(self, model, emb, z):
        """emb: (B, T, D) or (B, T, N, D) states, z: (B, T-1, z_dim) latent actions of the true transitions"""
        pred = model.rollout(emb[:, :1], z[:, :0], z[:, None])[:, 0]  # (B, T-1, ...)
        target = emb[:, 1:].float()
        self.err += (pred.float() - target).pow(2).flatten(2).mean(-1).sum(0)
        self.persistence += (emb[:, :1].float() - target).pow(2).flatten(2).mean(-1).sum(0)
        self.n += len(emb)

    def compute(self):
        err, pers = (self.err / self.n).tolist(), (self.persistence / self.n).tolist()
        out = {}
        for h, (e, p) in enumerate(zip(err, pers), start=1):
            out |= {f"error_h{h}": e, f"persistence_h{h}": p, f"skill_h{h}": 1 - e / p}
            if h > 1:
                out[f"growth_h{h}"] = e / err[h - 2]
        return out
