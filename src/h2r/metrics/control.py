"""Control: the world model driven by real actions through the controller, as at planning time"""

import torch
import torch.nn.functional as F

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]  # end-effector deltas (V-JEPA 2-AC), h2r data extract
DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]
GRIPPER = ["close", "open"]
REGIMES = ["still", "free", "gripper"]
PROBED = [0, 1, 2, 6]  # the end-effector state read back from the world model's states: x, y, z, gripper closedness
PROBED_NAMES = ["x", "y", "z", "gripper"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # categorical, fixed order


class ControlledRollout:
    """Open-loop unrolling from one real state (JEPA-WM's protocol, as RolloutError), with four sources of latent actions:
    the controller from the true real actions (ctrl), the IDM (idm: a privileged reference, it sees the next frame), the
    controller from another episode's actions (other) and one constant latent action, the mean of the training mu (mean).
    - skill_h{h}, skill_idm_h{h}, skill_mean_h{h}: 1 - error / persistence (copy of s_0) h steps ahead, the MSE skill
      score of forecast verification (Murphy 1988, Mon. Weather Rev. 116; persistence as the reference: Murphy 1992,
      Weather Forecast. 7);
    - lost_h{h} = skill_idm_h{h} - skill_h{h}: the gap between the IDM and this controller, with this representation and
      these data. It mixes the controller's capacity, what the current frame does not show, the domain shift and what
      the IDM gets from seeing the future (the leak of diagnostics E8): it does not isolate the leak. Not a bound: the
      IDM is a reference, not a ceiling. A difference, not a ratio: the oracle skill can be near 0 far from the
      training videos;
    - error_ratio_h{h}: error with the controller / error with the IDM, the measure of Garrido et al. 2026 (tables S1-S2:
      x1.0-1.5 on DROID and RECON, in LPIPS of decoded frames: not a threshold for this embedding MSE);
    - controllability_h{h}: 1 - error with the true actions / error with another episode's, 0 when the actions do not
      steer the rollout; Genie's controllability (Bruce et al. 2024, ΔPSNR between inferred and random latent actions)
      in embedding MSE, with real actions: controllability = 1 - 10^(-ΔPSNR / 10). The rollout is sensitive to which
      actions it gets; not a proof that it moves the right way;
    - skill and controllability: their means over the horizon.
    The oracle and the mean use no controller: skill_mean is a constant baseline (a poor controller can fall below it).
    With groups (regimes of the windows), also per regime and horizon: {regime}_n and the MSE {regime}_err_{source}_h{h}
    for ctrl, idm, mean and persistence (MSE rather than skill: persistence is ~0 on still windows).
    """

    def __init__(self, mean_z):
        """mean_z: (z_dim,) mean of the IDM's mu over the training transitions"""
        self.mean_z = mean_z
        self.err = {k: [] for k in ["ctrl", "idm", "other", "mean", "persistence"]}
        self.groups = []

    def update(self, model, ctrl, emb, mu, actions, groups=None):
        """emb: (B, H + 1, D) or (B, H + 1, N, D) states of an episode window, mu: (B, H, z_dim) the IDM's latent actions
        of its transitions, actions: (B, H, act_dim) the real ones, groups: (B,) the regime of each window (regimes)
        -> the predicted states of each source, (B, H, ...)"""
        start, past, target = emb[:, :1], mu[:, :0], emb[:, 1:].float()
        mean = self.mean_z.to(mu).expand_as(mu)
        preds = {
            "ctrl": model.rollout(start, past, actions[:, None], ctrl)[:, 0],
            "idm": model.rollout(start, past, mu[:, None])[:, 0],
            "other": model.rollout(start, past, actions.roll(1, 0)[:, None], ctrl)[:, 0],
            "mean": model.rollout(start, past, mean[:, None])[:, 0],
        }
        for k, pred in preds.items():
            self.err[k].append((pred.float() - target).pow(2).flatten(2).mean(-1).cpu())  # (B, H)
        self.err["persistence"].append((start.float() - target).pow(2).flatten(2).mean(-1).cpu())
        if groups is not None:
            self.groups.append(groups.cpu())
        return preds

    def compute(self):
        err = {k: torch.cat(v).double().mean(0) for k, v in self.err.items()}
        skill = {k: 1 - err[k] / err["persistence"] for k in ["ctrl", "idm", "mean"]}
        control = 1 - err["ctrl"] / err["other"]
        out = {"skill": float(skill["ctrl"].mean()), "controllability": float(control.mean())}
        for h in range(len(control)):
            out |= {
                f"skill_h{h + 1}": float(skill["ctrl"][h]),
                f"skill_idm_h{h + 1}": float(skill["idm"][h]),
                f"skill_mean_h{h + 1}": float(skill["mean"][h]),
                f"lost_h{h + 1}": float(skill["idm"][h] - skill["ctrl"][h]),
                f"error_ratio_h{h + 1}": float(err["ctrl"][h] / err["idm"][h]),
                f"controllability_h{h + 1}": float(control[h]),
            }
        if self.groups:
            groups = torch.cat(self.groups)
            for g, name in enumerate(REGIMES):
                sel = groups == g
                out[f"{name}_n"] = int(sel.sum())
                if not sel.any():
                    continue
                for k in ["ctrl", "idm", "mean", "persistence"]:
                    e = torch.cat(self.err[k])[sel].double().mean(0)
                    out |= {f"{name}_err_{k}_h{h + 1}": float(v) for h, v in enumerate(e)}
        return out


def regimes(actions, still_xyz=0.005, gripper_change=0.1):
    """The regime of each window from its real actions (W, H, act_dim): gripper (2) when the closedness changes by more
    than gripper_change at some step, else still (0) when every step moves the end effector by less than still_xyz
    (rotations not counted), else free motion (1). DROID has no contact sensor: contact is not a regime here"""
    gripper = (actions[..., -1].abs() > gripper_change).any(1)
    still = (actions[..., :3].norm(dim=-1) < still_xyz).all(1)
    return torch.where(gripper, 2, torch.where(still, 0, 1))


def ridge(x, y, rel=1e-3):
    """Linear least squares with a small ridge (rel x the mean diagonal of the Gram matrix), with a bias: the fitted
    function, (M, F) -> (M, targets)"""
    x1 = torch.cat([x.double(), torch.ones(len(x), 1, dtype=torch.double)], 1)
    gram = x1.T @ x1
    w = torch.linalg.solve(gram + rel * gram.diagonal().mean() * torch.eye(len(gram), dtype=torch.double),
                           x1.T @ y.double())
    return lambda z: torch.cat([z.double(), torch.ones(len(z), 1, dtype=torch.double)], 1) @ w


def r2(pred, y):
    """r2 per target dimension, None where the target is (almost) constant: [float | None]"""
    y = y.double()
    var = (y - y.mean(0)).pow(2).sum(0)
    out = 1 - (pred.double() - y).pow(2).sum(0) / var.clamp_min(1e-30)
    return [None if c else float(v) for v, c in zip(out, var <= 1e-10 * len(y))]


def mean_r2(values):
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def action_probe(mu_train, a_train, mu_test, a_test):
    """Does z encode the robot's motion? Ridge regression of the real action from the IDM's mu (the latent -> action
    decoder of LAPO, Schmidt & Jiang 2024, linear here), fitted on the training transitions, r2 on the test ones per
    action dimension (1: the action is read from z, 0: no better than its mean) and averaged. A dimension (almost)
    constant on the test transitions has no r2 (e.g. the rotations of the sim episodes, never commanded): left out of
    the mean, listed in "constant".
    mu: (M, z_dim), a: (M, act_dim)"""
    scores = r2(ridge(mu_train, a_train)(mu_test), a_test)
    names = ACTION_NAMES if len(scores) == len(ACTION_NAMES) else [str(i) for i in range(len(scores))]
    out = {"r2": mean_r2(scores), "constant": sum(v is None for v in scores)}
    return out | {f"r2_{n}": v for n, v in zip(names, scores)}


@torch.no_grad()
def plan(model, ctrl, context, past, goal, horizon, samples, elites, iterations, generator, init_std=1.0, bounds=None,
         trace=None):
    """The planner of Garrido et al. 2026 (appendix A, from Terver et al. 2025): the Cross-Entropy Method over sequences
    of real actions, each unrolled from the context states through the controller and the world model, cost = squared
    L2 of the last predicted state to the goal state. Sampled in standardized actions (the controller's statistics),
    from N(0, init_std) (Garrido et al.: zero mean, unit variance); bounds: (act_dim,) |action| per dimension in the
    dataset's units, the candidates clipped to them (Terver et al.'s DROID config: max_norms 0.1 on xyz and the
    rotations, 0.75 on the gripper), None = unbounded.
    context: (B, C, D) or (B, C, N, D) the last C real states, past: (B, C - 1, z_dim) the latent actions linking them,
    goal: (B, D) or (B, N, D) -> the best sequence of the last iteration, (B, horizon, act_dim).
    trace: a list, one dict per iteration appended: clipped (B,) share of the candidates with a dimension out of the
    bounds, clipped_dims (act_dim,) share of the candidate actions out of the bounds per dimension, best_cost (B,) the
    predicted cost of the best candidate, best (B, horizon, act_dim) that candidate"""
    b, act_dim = len(context), len(ctrl.act_mean)
    mean = torch.zeros(b, horizon, act_dim, device=context.device)
    std = torch.full_like(mean, init_std)
    for _ in range(iterations):
        noise = torch.randn(b, samples, horizon, act_dim, generator=generator).to(context.device)
        actions = ctrl.act_mean + ctrl.act_std * (mean[:, None] + std[:, None] * noise)  # (B, S, H, act_dim)
        out = torch.zeros_like(actions, dtype=torch.bool) if bounds is None else actions.abs() > bounds
        if bounds is not None:
            actions = torch.maximum(torch.minimum(actions, bounds), -bounds)
        pred = model.rollout(context, past, actions, ctrl)
        cost = (pred[:, :, -1].float() - goal[:, None].float()).pow(2).flatten(2).sum(-1)  # (B, S)
        best = cost.topk(elites, dim=1, largest=False).indices
        elite = torch.gather(actions, 1, best[..., None, None].expand(-1, -1, horizon, act_dim))
        u = (elite - ctrl.act_mean) / ctrl.act_std
        mean, std = u.mean(1), u.std(1)
        if trace is not None:
            trace.append({
                "clipped": out.flatten(2).any(-1).float().mean(1).cpu(),
                "clipped_dims": out.flatten(0, 2).float().mean(0).cpu(),
                "best_cost": cost.gather(1, best[:, :1])[:, 0].cpu(),
                "best": elite[:, 0].float().cpu(),
            })
    return elite[:, 0]


class PlanningError:
    """Offline planning to a goal frame (Garrido et al. 2026, protocol of Terver et al. 2025): from s_t of an episode, plan
    the horizon actions that lead to s_t+horizon, encoded from the episode's frame.
    - dxyz: L1 distance between the summed planned and true end-effector translations (x, y, z), in the dataset's units
      (m in robosuite and DROID); Garrido et al.: 0.10-0.49 m on DROID, V-JEPA 2-AC 0.15;
    - dxyz_still: the same for planning no motion (|summed true translation|_1), the level to beat;
    - dxyz_ratio = dxyz / dxyz_still: below 1, the planned displacement agrees better with the recorded one than no
      motion. Offline: the plan is compared to the demonstration, not executed."""

    def __init__(self):
        self.err, self.still, self.n = 0.0, 0.0, 0

    def update(self, planned, actions):
        """planned, actions: (B, horizon, act_dim) planned and true actions from the same start state"""
        true = actions[..., :3].sum(1)
        self.err += (planned[..., :3].sum(1) - true).abs().sum(-1).sum().item()
        self.still += true.abs().sum(-1).sum().item()
        self.n += len(actions)

    def compute(self):
        return {"dxyz": self.err / self.n, "dxyz_still": self.still / self.n, "dxyz_ratio": self.err / self.still}


def action_classes(a, std, gripper_change=0.1):
    """Two labels of each real action (M, act_dim), the gripper apart (it changes at many steps, it would dilute a
    seventh class): direction = the dominant translation axis and its sign, in stds of each axis (DIRECTIONS);
    gripper = closing (0) or opening (1) when its closedness changes by more than gripper_change, else -1 (left out)"""
    t = a[:, :3] / std[:3]
    axis = t.abs().argmax(1)
    direction = 2 * axis + (t.gather(1, axis[:, None])[:, 0] < 0).long()
    gripper = torch.full_like(direction, -1)
    gripper[a[:, -1] > gripper_change], gripper[a[:, -1] < -gripper_change] = 0, 1
    return {"direction": direction, "gripper": gripper}


def balanced_accuracy(pred, y, n):
    """Mean recall over the classes present, and the confusion matrix (rows: true class, normalized)"""
    conf = torch.zeros(n, n).index_put_((y, pred), torch.ones(len(y)), accumulate=True)
    present = conf.sum(1) > 0
    recall = conf.diagonal() / conf.sum(1).clamp_min(1)
    return recall[present].mean().item(), conf / conf.sum(1, keepdim=True).clamp_min(1)


def classify(xtr, ytr, xte, yte, n, k=10):
    """Balanced accuracy and confusion of a k-nearest-neighbours vote (Euclidean) and of a linear classifier (ridge on
    one-hot targets), features standardized on the training points: {"kNN": (acc, conf), "linear": (acc, conf)}"""
    m, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = ((x - m) / sd for x in (xtr, xte))
    near = torch.cat([torch.cdist(xte[i : i + 1024], xtr).topk(k, largest=False).indices for i in range(0, len(xte), 1024)])
    knn = torch.mode(ytr[near], 1).values
    x1 = torch.cat([xtr, torch.ones(len(xtr), 1)], 1).double()
    w = torch.linalg.solve(x1.T @ x1 + 1e-2 * len(x1) * torch.eye(x1.shape[1], dtype=torch.double),
                           x1.T @ F.one_hot(ytr, n).double())
    linear = (torch.cat([xte, torch.ones(len(xte), 1)], 1).double() @ w).argmax(1)
    return {"kNN": balanced_accuracy(knn, yte, n), "linear": balanced_accuracy(linear, yte, n)}


def class_probes(train, test, labels_train, labels_test):
    """Does each feature name the real action? (diagnostics E11 A) train / test: {feature: (M, F)} e.g. mu (the IDM),
    controller z, s_t, Δs = s_t+1 - s_t; labels: action_classes. Balanced accuracy of kNN and linear classifiers for the
    direction of the translation (chance 1/6) and the gripper closing / opening (chance 1/2).
    How to read: mu near Δs = the IDM keeps what the states show of the motion; Δs near chance = the encoder does not see
    it; s_t well above chance = the scene alone predicts the action (teleoperation: the arm's position gives the next
    move away), compare mu and the controller's z to it"""
    out, conf = {}, {}
    for task, names in [("direction", DIRECTIONS), ("gripper", GRIPPER)]:
        tr, te = labels_train[task] >= 0, labels_test[task] >= 0
        out[f"{task}_n_test"] = int(te.sum())
        if tr.sum() < 10 or te.sum() < 1 or len(labels_train[task][tr].unique()) < 2:
            continue
        for name in train:
            r = classify(train[name][tr], labels_train[task][tr], test[name][te], labels_test[task][te], len(names))
            for clf, (acc, c) in r.items():
                out[f"{task}_{name}_{clf}"] = acc
                conf[task, name, clf] = c
    return out, conf


@torch.no_grad()
def counterfactual(model, ctrl, s, true_step, spread, generator, n=2000, ks=(-2, -1, 1, 2)):
    """One action dimension at a time through the controller (diagnostics E11 B). From n real states s (M, D), the mean
    training action, then the same action with one dimension moved by k stds of the training actions:
    Δz = controller(moved action, s) - controller(mean action, s), and Δprediction = |predict(s, z + Δz) - predict(s, z)|.
    Per dimension, at ±2 stds: dz_over_state (|Δz| against the change of z when the mean action comes from another
    state, 1 = the action moves z as much as the scene does), dz_over_spread (against the spread of the IDM's mu),
    consistency (median cos of each Δz with their mean, 1 = the same direction from every state), opposite (cos of the
    mean Δz at +2 and -2 stds, -1 = the two signs push z opposite ways), dpred_over_step (the change of the predicted
    next state against the median real step). separability: kNN and linear balanced accuracy of the 2 x act_dim cases
    from Δz, half the states to fit, half to test (chance 1 / (2 act_dim)).
    Caveat: the data may have no pure single-dimension action (rotations come with translations), so one dimension
    alone can be off the training distribution. Returns (metrics, the Δz at ±2 stds and their case, for the figure)"""
    idx = torch.randperm(len(s), generator=generator)[:n]
    s = s[idx].to(ctrl.act_mean.device).float()
    n, flat = len(s), s.flatten(1)
    a0 = ctrl.act_mean.expand(n, -1)
    z0 = ctrl(a0, flat)
    p0 = model.predict(s[:, None], z0[:, None])[:, 0]
    state_effect = (z0 - z0.roll(1, 0)).norm(dim=-1).median()
    out, dz, cases, means = {}, [], [], {}
    names = ACTION_NAMES if len(ctrl.act_mean) == len(ACTION_NAMES) else [str(i) for i in range(len(ctrl.act_mean))]
    for i, name in enumerate(names):
        rows = {}
        for k in ks:
            a = a0.clone()
            a[:, i] += k * ctrl.act_std[i]
            d = ctrl(a, flat) - z0
            dp = (model.predict(s[:, None], (z0 + d)[:, None])[:, 0] - p0).flatten(1).norm(dim=-1)
            rows[k] = {"dz_over_state": (d.norm(dim=-1).median() / state_effect).item(),
                       "dz_over_spread": (d.norm(dim=-1).median() / spread).item(),
                       "consistency": F.cosine_similarity(d, d.mean(0, keepdim=True), dim=-1).median().item(),
                       "dpred_over_step": (dp.median() / true_step).item()}
            means[name, k] = d.mean(0)
            if abs(k) == 2:
                dz.append(d.float().cpu())
                cases.append(torch.full((n,), 2 * i + int(k < 0)))
        for key in rows[2]:
            out[f"{name}_{key}"] = (rows[2][key] + rows[-2][key]) / 2
        out[f"{name}_opposite"] = F.cosine_similarity(means[name, 2], means[name, -2], dim=0).item()
    dz, cases = torch.cat(dz), torch.cat(cases)
    half = (torch.arange(len(cases)) % n) < n // 2
    sep = classify(dz[half], cases[half], dz[~half], cases[~half], 2 * len(names))
    out |= {"separability_kNN": sep["kNN"][0], "separability_linear": sep["linear"][0],
            "state_effect_over_spread": (state_effect / spread).item()}
    return out, (dz, cases, names)


def pca2(x):
    """First two principal components of (M, F)"""
    x = x.float() - x.float().mean(0)
    return x @ torch.linalg.svd(x, full_matrices=False).Vh[:2].T


def pca_basis(x, k=2):
    """Mean and first k principal axes of (M, F), to project other points the same way: (x - mean) @ axes"""
    mean = x.float().mean(0)
    return mean, torch.linalg.svd(x.float() - mean, full_matrices=False).Vh[:k].T


def spectrum(x):
    """Share of the variance along each principal axis of (M, F), decreasing; effective rank = exp(entropy of the
    shares) (Roy & Vetterli 2007), dims_90 = axes holding 90 % of the variance"""
    x = x.double() - x.double().mean(0)
    share = torch.linalg.svdvals(x).pow(2)
    share = share / share.sum()
    rank = torch.exp(-(share * share.clamp_min(1e-30).log()).sum())
    return share.float(), {"effective_rank": float(rank), "dims_90": int((share.cumsum(0) < 0.9).sum()) + 1}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "font.size": 9})
    return plt


def fig_actions(actions, bounds):
    """The real actions of the training transitions, one histogram per dimension (log counts), the planner's bounds"""
    plt = _plt()
    names = ACTION_NAMES if actions.size(1) == len(ACTION_NAMES) else [str(i) for i in range(actions.size(1))]
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3), layout="constrained")
    for i, (ax, name) in enumerate(zip(axs, names)):
        ax.hist(actions[:, i].numpy(), bins=80, color=SERIES[0], log=True)
        if bounds is not None:
            for b in (-bounds[i].item(), bounds[i].item()):
                ax.axvline(b, color="#52514e", ls=":", lw=1.2)
        ax.set(title=f"{name} (std {actions[:, i].std():.3g})", xlabel="per step")
    fig.suptitle("Real actions of the training transitions (end-effector deltas per 250 ms); dotted: planner bounds")
    return fig


def fig_latent_actions(features, labels, conf):
    """PCA of mu and of the controller's z of the test transitions, coloured by the direction of the translation
    (diagnostics E11 A), with the confusion of the directions from mu (kNN)"""
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, name in zip(axs[:2], ["mu", "controller_z"]):
        p = pca2(features[name]).numpy()
        for i, c in enumerate(DIRECTIONS):
            sel = (labels["direction"] == i).numpy()
            ax.scatter(p[sel, 0], p[sel, 1], s=3, alpha=0.4, color=SERIES[i], label=c)
        ax.set(title=f"PCA of {name}, colour = direction of the translation", xticks=[], yticks=[])
    axs[0].legend(fontsize=7, markerscale=4)
    c = conf.get(("direction", "mu", "kNN"))
    if c is not None:
        axs[2].imshow(c.numpy(), vmin=0, vmax=1, cmap="Blues")
        axs[2].set_xticks(range(len(DIRECTIONS)), DIRECTIONS)
        axs[2].set_yticks(range(len(DIRECTIONS)), DIRECTIONS)
        axs[2].grid(False)
        axs[2].set(title="direction from mu, kNN: confusion (rows: true)", xlabel="predicted", ylabel="true")
    fig.suptitle("Test transitions: are the latent actions organized by the real action?")
    return fig


def fig_counterfactual(dz, cases, dims):
    """PCA of Δz, one colour per action dimension, filled +2 stds, hollow -2 stds (E11 B)"""
    plt = _plt()
    p, cases = pca2(dz).numpy(), cases.numpy()
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for i, name in enumerate(dims):
        for sign, face in [(0, SERIES[i]), (1, "none")]:
            sel = cases == 2 * i + sign
            ax.scatter(p[sel, 0], p[sel, 1], s=8, facecolors=face, edgecolors=SERIES[i], lw=0.6, alpha=0.5,
                       label=f"{'+' if sign == 0 else '-'}{name}")
    ax.legend(fontsize=6, ncol=2, markerscale=2)
    ax.set(title="PCA of Δz: one action dimension moved by ±2 stds", xticks=[], yticks=[])
    return fig


def fig_regimes(metrics, horizon):
    """MSE per regime and horizon, one panel per source of latent actions, one colour scale for all: where and when
    the rollout degrades (the persistence panel: how much the scene actually changes)"""
    plt = _plt()
    sources = ["ctrl", "idm", "mean", "persistence"]
    grid = torch.full((len(sources), len(REGIMES), horizon), float("nan"))
    for s, k in enumerate(sources):
        for g, name in enumerate(REGIMES):
            for h in range(horizon):
                grid[s, g, h] = metrics.get(f"{name}_err_{k}_h{h + 1}", float("nan"))
    vmax = float(grid[~grid.isnan()].max()) if (~grid.isnan()).any() else 1.0
    fig, axs = plt.subplots(1, len(sources), figsize=(4 * len(sources), 2.6), layout="constrained", sharey=True)
    for ax, k, g in zip(axs, sources, grid):
        im = ax.imshow(g.numpy(), vmin=0, vmax=vmax, cmap="Blues", aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(horizon), [str(h + 1) for h in range(horizon)])
        ax.set_yticks(range(len(REGIMES)), [f"{r} ({metrics.get(f'{r}_n', 0)})" for r in REGIMES])
        ax.set(title=k, xlabel="horizon (steps of 250 ms)")
    fig.colorbar(im, ax=axs, label="MSE to the true states", shrink=0.9)
    fig.suptitle("Open-loop error by regime of the test windows and horizon")
    return fig


def fig_state_probe(probe):
    """r2 of the end-effector state decoded from the states: real (the probe's own level), predicted by each source h
    steps ahead, and the persistence s_0"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    hs = list(range(1, len(probe["ctrl"]) + 1))
    for i, k in enumerate(["ctrl", "idm", "mean", "persistence"]):
        ax.plot(hs, probe[k], color=SERIES[i], lw=2, marker="o", ms=4, label=k)
    if probe["real"] is not None:
        ax.axhline(probe["real"], color="#52514e", ls=":", lw=1.2, label="real states")
    ax.legend(fontsize=8)
    ax.set(title="End-effector x, y, z, gripper decoded from the states (linear probe)", xlabel="horizon",
           ylabel="r2 on the test windows (mean of the dimensions)", xticks=hs)
    return fig


def fig_spectra(spectra):
    """Share of the variance per principal axis (log), states and changes of state, real and imagined"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for i, (name, (share, stats)) in enumerate(spectra.items()):
        ax.plot(range(1, len(share) + 1), share.numpy(), color=SERIES[i], lw=2,
                label=f"{name} (eff. rank {stats['effective_rank']:.1f}, 90 %: {stats['dims_90']} axes)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set(title="Centered spectra: how many directions do the states and the motion use?", xlabel="principal axis",
           ylabel="share of the variance", xlim=(0, 64))
    return fig


def fig_trajectories(background, real, ctrl, idm):
    """PCA of the states fitted on real training states (grey), the same projection for a few test windows: real
    trajectory, rollout with the controller, rollout with the IDM. background: (M, 2), real / ctrl / idm: (W, H + 1, 2)"""
    plt = _plt()
    n = len(real)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7), layout="constrained")
    for w, ax in enumerate(axs.flat):
        if w >= n:
            ax.axis("off")
            continue
        ax.scatter(background[:, 0], background[:, 1], s=1, color="#c3c2b7", alpha=0.3)
        for i, (name, traj) in enumerate([("real", real), ("controller", ctrl), ("IDM", idm)]):
            ax.plot(traj[w, :, 0], traj[w, :, 1], color=SERIES[i], lw=2, marker="o", ms=3, label=name)
        ax.scatter(real[w, 0, 0], real[w, 0, 1], s=40, color="#0b0b0b", zorder=3)
        ax.set(xticks=[], yticks=[], title=f"test window {w + 1}")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle("Real and imagined trajectories in the PCA of the real states (black dot: start)")
    return fig


def fig_cem(trace):
    """Per CEM iteration, mean over the planning clips: share of candidates clipped to the bounds, predicted cost of the
    best candidate, and its translation's distance to the demonstration (dxyz; dotted: planning no motion)"""
    plt = _plt()
    its = list(range(1, len(trace["clipped"]) + 1))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), layout="constrained")
    for ax, key, title in [(axs[0], "clipped", "candidates out of the bounds (share)"),
                           (axs[1], "best_cost", "predicted cost of the best candidate"),
                           (axs[2], "dxyz", "dxyz of the best candidate to the demonstration")]:
        ax.plot(its, trace[key], color=SERIES[0], lw=2, marker="o", ms=4)
        ax.set(title=title, xlabel="CEM iteration")
    axs[2].axhline(trace["dxyz_still"], color="#52514e", ls=":", lw=1.2, label="no motion")
    axs[2].legend(fontsize=8)
    fig.suptitle("Does CEM improve the plan? (offline: dxyz against the recorded actions, not an execution)")
    return fig
