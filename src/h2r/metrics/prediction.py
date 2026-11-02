"""Prediction quality: one step with the true past (teacher forcing) and open-loop unrolling, against the trivial
prediction"""

import torch
import torch.nn.functional as F

from h2r.metrics.stats import Moments


class PredictionQuality:
    """
    - tf: MSE of the next state with the IDM's latent actions, over every transition of the clip;
    - persistence: MSE of the trivial prediction s_t+1 = s_t on the same transitions, the level to beat;
    - std_ratio: std of the predictions / std of the targets, near 0 when the predictor outputs the mean;
    - tf_over_drift: error over FPP3's drift method (s_t + the clip's mean change so far), a second naive benchmark
      that moving clips make harder to beat than persistence;
    - change_ratio: median of |pred - s_t| / |s_t+1 - s_t|, above 1 = invented motion (1.8 on static clips,
      diagnostics E2-E5), below 1 = shrunk motion (0.5 when the KL starves z, E3);
    - change_cos: median cosine between the predicted change pred - s_t and the true one s_t+1 - s_t (direction);
    - retrieval: share of the transitions whose prediction is closer to s_t+1 than to every other frame of the clip
      (chance 1 / T; E5-E6);
    - the same per third of the transitions with the least (static), middle and most (moving) pixel change (mean
      absolute change between the two frames, cut at the thirds of the whole validation set: the same split for every
      run and every encoder), with tf, persistence and tf_over_persistence: the static half of the data hides the
      moving one in the means (diagnostics E1-E3: tf / persistence 2.6 static, 0.33 moving);
    - invented_static: median |pred - s_t| on the static third over the median |s_t+1 - s_t| of all transitions. The
      invented motion with a denominator that does not vanish: change_ratio_static divides by the true change of
      nearly still transitions (x2.5 on the most static decile for every net of E12). true_change_static: the same
      for the true change, the level a right-sized prediction would reach.
    """

    def __init__(self):
        self.preds, self.targets = Moments(), Moments()
        self.n, self.err, self.persistence = 0, 0.0, 0.0
        keys = ["motion", "err", "persistence", "drift", "change_pred", "change_true", "change_cos", "retrieval"]
        self.raw = {k: [] for k in keys}

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
        t = torch.arange(e.size(1) - 1, device=e.device).clamp_min(1)[None, :, None]
        drift = e[:, :-1] + (e[:, :-1] - e[:, :1]) / t  # persistence at t = 0
        frames = torch.arange(1, e.size(1), device=e.device)
        new = {
            "motion": (clip[:, 1:].float() - clip[:, :-1].float()).abs().mean((2, 3, 4)),
            "err": (p - e[:, 1:]).pow(2).mean(-1),
            "persistence": d_true.pow(2).mean(-1),
            "drift": (drift - e[:, 1:]).pow(2).mean(-1),
            "change_pred": d_pred.norm(dim=-1),
            "change_true": d_true.norm(dim=-1),
            "change_cos": F.cosine_similarity(d_pred, d_true, dim=-1),
            "retrieval": (torch.cdist(p, e).argmin(-1) == frames).float(),
        }
        for k, v in new.items():
            self.raw[k].append(v.flatten().cpu())

    def compute(self):
        ratio = (self.preds.var().sum() / self.targets.var().sum()).sqrt()
        r = {k: torch.cat(v) for k, v in self.raw.items()}
        low, high = r["motion"].quantile(torch.tensor([1 / 3, 2 / 3]))
        groups = {
            "": torch.ones_like(r["motion"], dtype=torch.bool),
            "_static": r["motion"] <= low,
            "_middle": (r["motion"] > low) & (r["motion"] < high),
            "_moving": r["motion"] >= high,
        }
        out = {
            "tf": float(self.err / self.n),
            "persistence": float(self.persistence / self.n),
            "tf_over_persistence": float(self.err / self.persistence),
            "std_ratio": float(ratio),
        }
        for name, keep in groups.items():
            err, pers = r["err"][keep].mean(), r["persistence"][keep].mean()
            if name:
                out |= {f"tf{name}": float(err), f"persistence{name}": float(pers),
                        f"tf_over_persistence{name}": float(err / pers)}
            out |= {
                f"tf_over_drift{name}": float(err / r["drift"][keep].mean()),
                f"change_ratio{name}": float((r["change_pred"][keep] / r["change_true"][keep].clamp_min(1e-8)).median()),
                f"change_cos{name}": float(r["change_cos"][keep].median()),
                f"retrieval{name}": float(r["retrieval"][keep].mean()),
            }
        typical = r["change_true"].median()
        out["invented_static"] = float(r["change_pred"][groups["_static"]].median() / typical)
        out["true_change_static"] = float(r["change_true"][groups["_static"]].median() / typical)
        return out


class PredictorInternals:
    """
    - gate_msa, gate_mlp: mean |gate| of the AdaLN-zero blocks of the predictor (0 at initialization: the blocks add
      nothing and the prediction is the projected input);
    - blocks_share: |x_L - x_0| / |x_0| in the predictor's residual stream, what its blocks add to the input states.
    """

    def __init__(self):
        self.n, self.msa, self.mlp, self.share = 0, 0.0, 0.0, 0.0

    def update(self, model, emb, z):
        t = model.predictor.transformer
        x, gates = [], []
        hooks = [t.layers[0].register_forward_pre_hook(lambda m, args: x.append(args[0].float())),
                 t.norm.register_forward_pre_hook(lambda m, args: x.append(args[0].float()))]
        hooks += [b.adaLN_modulation.register_forward_hook(lambda m, i, o: gates.append(o.float().chunk(6, -1)))
                  for b in t.layers]
        try:
            model.predict(emb[:, :-1], z)
        finally:
            for h in hooks:
                h.remove()
        b = len(emb)
        self.msa += torch.stack([g[2].abs().mean() for g in gates]).mean() * b
        self.mlp += torch.stack([g[5].abs().mean() for g in gates]).mean() * b
        self.share += (x[1] - x[0]).norm() / x[0].norm() * b
        self.n += b

    def compute(self):
        return {k: float(v / self.n) for k, v in
                {"gate_msa": self.msa, "gate_mlp": self.mlp, "blocks_share": self.share}.items()}


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
