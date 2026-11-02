"""Control: the world model driven by real actions through the controller, as at planning time"""

import torch
import torch.nn.functional as F

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]  # end-effector deltas (V-JEPA 2-AC), h2r data extract
DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]
GRIPPER = ["close", "open"]


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
    """

    def __init__(self, mean_z):
        """mean_z: (z_dim,) mean of the IDM's mu over the training transitions"""
        self.mean_z = mean_z
        self.n, self.persistence = 0, 0.0
        self.err = {k: 0.0 for k in ["ctrl", "idm", "other", "mean"]}

    def update(self, model, ctrl, emb, mu, actions):
        """emb: (B, H + 1, D) or (B, H + 1, N, D) states of an episode window, mu: (B, H, z_dim) the IDM's latent actions
        of its transitions, actions: (B, H, act_dim) the real ones"""
        start, past, target = emb[:, :1], mu[:, :0], emb[:, 1:].float()
        mean = self.mean_z.to(mu).expand_as(mu)
        preds = {
            "ctrl": model.rollout(start, past, actions[:, None], ctrl),
            "idm": model.rollout(start, past, mu[:, None]),
            "other": model.rollout(start, past, actions.roll(1, 0)[:, None], ctrl),
            "mean": model.rollout(start, past, mean[:, None]),
        }
        for k, pred in preds.items():
            self.err[k] += (pred[:, 0].float() - target).pow(2).flatten(2).mean(-1).sum(0)
        self.persistence += (start.float() - target).pow(2).flatten(2).mean(-1).sum(0)
        self.n += len(emb)

    def compute(self):
        pers = self.persistence / self.n
        skill = {k: (1 - v / self.n / pers) for k, v in self.err.items()}
        control = 1 - self.err["ctrl"] / self.err["other"]
        out = {"skill": float(skill["ctrl"].mean()), "controllability": float(control.mean())}
        for h in range(len(pers)):
            out |= {
                f"skill_h{h + 1}": float(skill["ctrl"][h]),
                f"skill_idm_h{h + 1}": float(skill["idm"][h]),
                f"skill_mean_h{h + 1}": float(skill["mean"][h]),
                f"lost_h{h + 1}": float(skill["idm"][h] - skill["ctrl"][h]),
                f"error_ratio_h{h + 1}": float(self.err["ctrl"][h] / self.err["idm"][h]),
                f"controllability_h{h + 1}": float(control[h]),
            }
        return out


def action_probe(mu_train, a_train, mu_test, a_test):
    """Does z encode the robot's motion? Ridge regression of the real action from the IDM's mu (the latent -> action
    decoder of LAPO, Schmidt & Jiang 2024, linear here), fitted on the training transitions, r2 on the test ones per
    action dimension (1: the action is read from z, 0: no better than its mean) and averaged. A dimension (almost)
    constant on the test transitions has no r2 (e.g. the rotations of the sim episodes, never commanded): left out of
    the mean, listed in "constant".
    mu: (M, z_dim), a: (M, act_dim)"""
    x_train, x_test = (torch.cat([m.double(), torch.ones(len(m), 1, dtype=torch.double)], 1) for m in (mu_train, mu_test))
    gram = x_train.T @ x_train
    ridge = 1e-3 * gram.diagonal().mean() * torch.eye(len(gram), dtype=torch.double)
    w = torch.linalg.solve(gram + ridge, x_train.T @ a_train.double())
    a_test = a_test.double()
    var = (a_test - a_test.mean(0)).pow(2).sum(0)
    constant = var <= 1e-10 * len(a_test)
    r2 = 1 - (x_test @ w - a_test).pow(2).sum(0) / var.clamp_min(1e-30)
    names = ACTION_NAMES if len(r2) == len(ACTION_NAMES) else [str(i) for i in range(len(r2))]
    out = {"r2": float(r2[~constant].mean()) if (~constant).any() else None, "constant": int(constant.sum())}
    return out | {f"r2_{n}": None if c else float(v) for n, v, c in zip(names, r2, constant)}


@torch.no_grad()
def plan(model, ctrl, context, past, goal, horizon, samples, elites, iterations, generator, init_std=1.0, bounds=None):
    """The planner of Garrido et al. 2026 (appendix A, from Terver et al. 2025): the Cross-Entropy Method over sequences
    of real actions, each unrolled from the context states through the controller and the world model, cost = squared
    L2 of the last predicted state to the goal state. Sampled in standardized actions (the controller's statistics),
    from N(0, init_std) (Garrido et al.: zero mean, unit variance); bounds: (act_dim,) |action| per dimension in the
    dataset's units, the candidates clipped to them (Terver et al.'s DROID config: max_norms 0.1 on xyz and the
    rotations, 0.75 on the gripper), None = unbounded.
    context: (B, C, D) or (B, C, N, D) the last C real states, past: (B, C - 1, z_dim) the latent actions linking them,
    goal: (B, D) or (B, N, D) -> the best sequence of the last iteration, (B, horizon, act_dim)"""
    b, act_dim = len(context), len(ctrl.act_mean)
    mean = torch.zeros(b, horizon, act_dim, device=context.device)
    std = torch.full_like(mean, init_std)
    for _ in range(iterations):
        noise = torch.randn(b, samples, horizon, act_dim, generator=generator).to(context.device)
        actions = ctrl.act_mean + ctrl.act_std * (mean[:, None] + std[:, None] * noise)  # (B, S, H, act_dim)
        if bounds is not None:
            actions = torch.maximum(torch.minimum(actions, bounds), -bounds)
        pred = model.rollout(context, past, actions, ctrl)
        cost = (pred[:, :, -1].float() - goal[:, None].float()).pow(2).flatten(2).sum(-1)  # (B, S)
        best = cost.topk(elites, dim=1, largest=False).indices
        elite = torch.gather(actions, 1, best[..., None, None].expand(-1, -1, horizon, act_dim))
        u = (elite - ctrl.act_mean) / ctrl.act_std
        mean, std = u.mean(1), u.std(1)
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


def figures(actions, bounds, features, labels, conf, cf):
    """The figures of the controller's test: {name: matplotlib figure}
    - actions: the real actions of the training transitions, one histogram per dimension (log counts), the planner's
      bounds as dotted lines;
    - latent_actions: PCA of mu and of the controller's z of the test transitions, coloured by the direction of the
      translation (diagnostics E11 A), with the confusion of the directions from mu (kNN);
    - counterfactual: PCA of Δz, one colour per action dimension, filled +2 stds, hollow -2 stds (E11 B)"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figs = {}
    names = ACTION_NAMES if actions.size(1) == len(ACTION_NAMES) else [str(i) for i in range(actions.size(1))]
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3), layout="constrained")
    for i, (ax, name) in enumerate(zip(axs, names)):
        ax.hist(actions[:, i].numpy(), bins=80, color="C0", log=True)
        if bounds is not None:
            for b in (-bounds[i].item(), bounds[i].item()):
                ax.axvline(b, color="k", ls=":", lw=1)
        ax.set(title=f"{name} (std {actions[:, i].std():.3g})", xlabel="per step")
    fig.suptitle("Real actions of the training transitions (end-effector deltas per 250 ms); dotted: planner bounds")
    figs["actions"] = fig

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, name in zip(axs[:2], ["mu", "controller_z"]):
        p = pca2(features[name]).numpy()
        for i, c in enumerate(DIRECTIONS):
            sel = (labels["direction"] == i).numpy()
            ax.scatter(p[sel, 0], p[sel, 1], s=2, alpha=0.4, color=plt.cm.tab10(i), label=c)
        ax.set(title=f"PCA of {name}, colour = direction of the translation", xticks=[], yticks=[])
    axs[0].legend(fontsize=7, markerscale=5)
    c = conf.get(("direction", "mu", "kNN"))
    if c is not None:
        axs[2].imshow(c.numpy(), vmin=0, vmax=1, cmap="viridis")
        axs[2].set_xticks(range(len(DIRECTIONS)), DIRECTIONS)
        axs[2].set_yticks(range(len(DIRECTIONS)), DIRECTIONS)
        axs[2].set(title="direction from mu, kNN: confusion (rows: true)", xlabel="predicted", ylabel="true")
    fig.suptitle("Test transitions: are the latent actions organized by the real action?")
    figs["latent_actions"] = fig

    if cf is not None:
        dz, cases, dims = cf
        p, cases = pca2(dz).numpy(), cases.numpy()
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        for i, name in enumerate(dims):
            for sign, face in [(0, plt.cm.tab10(i)), (1, "none")]:
                sel = cases == 2 * i + sign
                ax.scatter(p[sel, 0], p[sel, 1], s=6, facecolors=face, edgecolors=plt.cm.tab10(i), lw=0.5, alpha=0.5,
                           label=f"{'+' if sign == 0 else '-'}{name}")
        ax.legend(fontsize=6, ncol=2, markerscale=3)
        ax.set(title="PCA of Δz: one action dimension moved by ±2 stds", xticks=[], yticks=[])
        figs["counterfactual"] = fig
    return figs
