"""Control: the world model driven by real actions through the controller, as at planning time"""

import torch

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]  # end-effector deltas (V-JEPA 2-AC), h2r data extract


class ControlledRollout:
    """Open-loop unrolling from one real state (JEPA-WM's protocol, as RolloutError), with four sources of latent actions:
    the controller from the true real actions (ctrl), the IDM (idm: an oracle, it sees the next frame), the controller
    from another episode's actions (other) and one constant latent action, the mean of the training mu (mean).
    - skill_h{h}, skill_idm_h{h}, skill_mean_h{h}: 1 - error / persistence (copy of s_0) h steps ahead, the MSE skill
      score of forecast verification (Murphy 1988, Mon. Weather Rev. 116; persistence as the reference: Murphy 1992,
      Weather Forecast. 7);
    - lost_h{h} = skill_idm_h{h} - skill_h{h}: the oracle skill that real actions do not reach. The skill a predictor
      gets from a z that copies the next frame (the leak of diagnostics E8) is lost: the controller only sees s_t and
      a_t. A difference, not a ratio: the oracle skill can be near 0 on frames far from the training videos;
    - error_ratio_h{h}: error with the controller / error with the IDM, the measure of Garrido et al. 2026 (tables S1-S2:
      x1.0-1.5 on DROID and RECON, in LPIPS of decoded frames);
    - controllability_h{h}: 1 - error with the true actions / error with another episode's, 0 when the actions do not
      steer the rollout; Genie's controllability (Bruce et al. 2024, ΔPSNR between inferred and random latent actions)
      in embedding MSE, with real actions: controllability = 1 - 10^(-ΔPSNR / 10);
    - skill and controllability: their means over the horizon.
    The oracle and the mean use no controller: skill_mean is the level reached without knowing the action.
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
    action dimension (1: the action is read from z, 0: no better than its mean) and averaged.
    mu: (M, z_dim), a: (M, act_dim)"""
    x_train, x_test = (torch.cat([m.double(), torch.ones(len(m), 1, dtype=torch.double)], 1) for m in (mu_train, mu_test))
    gram = x_train.T @ x_train
    ridge = 1e-3 * gram.diagonal().mean() * torch.eye(len(gram), dtype=torch.double)
    w = torch.linalg.solve(gram + ridge, x_train.T @ a_train.double())
    a_test = a_test.double()
    r2 = 1 - (x_test @ w - a_test).pow(2).sum(0) / (a_test - a_test.mean(0)).pow(2).sum(0)
    names = ACTION_NAMES if len(r2) == len(ACTION_NAMES) else [str(i) for i in range(len(r2))]
    return {"r2": float(r2.mean())} | {f"r2_{n}": float(v) for n, v in zip(names, r2)}


@torch.no_grad()
def plan(model, ctrl, start, goal, horizon, samples, elites, iterations, generator):
    """The planner of Garrido et al. 2026 (appendix A, from Terver et al. 2025): the Cross-Entropy Method over sequences
    of real actions, each unrolled from the start state through the controller and the world model, cost = squared L2
    of the last predicted state to the goal state. Standardized actions (the controller's statistics), from N(0, 1).
    start, goal: (B, D) or (B, N, D) -> the best sequence of the last iteration, (B, horizon, act_dim)"""
    b, act_dim = len(start), len(ctrl.act_mean)
    mean = torch.zeros(b, horizon, act_dim, device=start.device)
    std = torch.ones_like(mean)
    for _ in range(iterations):
        noise = torch.randn(b, samples, horizon, act_dim, generator=generator).to(start.device)
        actions = ctrl.act_mean + ctrl.act_std * (mean[:, None] + std[:, None] * noise)  # (B, S, H, act_dim)
        pred = model.rollout(start[:, None], start.new_zeros(b, 0, ctrl.head.out_features), actions, ctrl)
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
      (m in robosuite); Garrido et al.: 0.10-0.49 m on DROID, V-JEPA 2-AC 0.15;
    - dxyz_still: the same for planning no motion (|summed true translation|_1), the level to beat;
    - dxyz_ratio = dxyz / dxyz_still, below 1 = the plan moves the arm the right way."""

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
