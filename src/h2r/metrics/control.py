"""Control: the world model driven by real actions through the controller, the measures of Garrido et al. 2026
(section 5, appendix A, tables S1-S2)"""

import torch


class ControlledRollout:
    """Open-loop unrolling from one real state (JEPA-WM's protocol, as RolloutError), with the latent actions of the
    controller from the true real actions (ctrl) and of the IDM (idm, a reference: it sees the next frame).
    - error_h{h}, error_idm_h{h}: MSE of the state h steps ahead;
    - error_ratio_h{h}: error with the controller / error with the IDM, the measure of Garrido et al. 2026 (tables S1-S2:
      x1.0-1.5 on DROID and RECON, in LPIPS of decoded frames: not a threshold for this embedding MSE)."""

    def __init__(self):
        self.err = {"ctrl": [], "idm": []}

    def update(self, model, ctrl, emb, mu, actions):
        """emb: (B, H + 1, D) or (B, H + 1, N, D) states of an episode window, mu: (B, H, z_dim) the IDM's latent actions
        of its transitions, actions: (B, H, act_dim) the real ones"""
        start, past, target = emb[:, :1], mu[:, :0], emb[:, 1:].float()
        preds = {
            "ctrl": model.rollout(start, past, actions[:, None], ctrl)[:, 0],
            "idm": model.rollout(start, past, mu[:, None])[:, 0],
        }
        for k, pred in preds.items():
            self.err[k].append((pred.float() - target).pow(2).flatten(2).mean(-1).cpu())  # (B, H)

    def compute(self):
        err = {k: torch.cat(v).double().mean(0) for k, v in self.err.items()}
        out = {}
        for h in range(len(err["ctrl"])):
            out |= {
                f"error_h{h + 1}": float(err["ctrl"][h]),
                f"error_idm_h{h + 1}": float(err["idm"][h]),
                f"error_ratio_h{h + 1}": float(err["ctrl"][h] / err["idm"][h]),
            }
        return out


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
      (m in robosuite and DROID); Garrido et al.: 0.10-0.49 m on DROID, V-JEPA 2-AC 0.15. Offline: the plan is compared
      to the demonstration, not executed."""

    def __init__(self):
        self.err, self.n = 0.0, 0

    def update(self, planned, actions):
        """planned, actions: (B, horizon, act_dim) planned and true actions from the same start state"""
        self.err += (planned[..., :3].sum(1) - actions[..., :3].sum(1)).abs().sum(-1).sum().item()
        self.n += len(actions)

    def compute(self):
        return {"dxyz": self.err / self.n}
