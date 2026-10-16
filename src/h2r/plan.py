"""Plan with a trained controller: CEM over real actions, scored by JEPA.get_cost. The closed loop in robosuite is
h2r.eval.pick (PlannerPolicy).

The context holds the last real frames and the latent actions linking them: the controller on the executed (measured)
actions, as for the future, or the IDM on the real pairs (past_latents: idm).
"""

from pathlib import Path

import numpy as np
import torch
import yaml

from h2r.models.controller import build_controller
from h2r.models.jepa import load_world_model


def load(controller_run, device):
    """Controller of a run and the frozen world model it was trained for"""
    run = Path(controller_run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model, wm_cfg = load_world_model(cfg["world_model"], device)
    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.load_state_dict(torch.load(run / "best.pt", map_location=device)["model"])
    return model, ctrl.eval().requires_grad_(False)


@torch.no_grad()
def encode(model, frames, device):
    """(T, H, W, 3) uint8 -> (T, N, D)"""
    return model.encode(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(device))[0]


@torch.no_grad()
def context_latents(model, ctrl, emb, actions, source):
    """Latent actions between the context states: emb (C, N, D), actions (C - 1, act_dim) -> (C - 1, d_a)"""
    if len(emb) < 2:
        return emb.new_zeros(0, ctrl.head.out_features)
    if source == "idm":
        return model.idm(emb[None])[0][0]
    return ctrl(actions, emb[:-1])


def physical(u, ctrl, cfg):
    """Standardized actions -> physical units; dimensions not searched (rotations etc.) stay at 0"""
    searched = torch.zeros(u.size(-1), dtype=torch.bool, device=u.device)
    searched[cfg["searched"]] = True
    a = (u * ctrl.act_std + ctrl.act_mean) * searched
    if cfg.get("max_norm"):
        a[..., :3] *= (cfg["max_norm"] / a[..., :3].norm(dim=-1, keepdim=True)).clamp(max=1)
    return a


@torch.no_grad()
def plan(model, ctrl, emb, latents, goal, cfg, generator=None):
    """CEM in the controller's standardized action space (mean 0, std 1 at the start), searched dimensions only.
    Returns (H, act_dim) actions in physical units."""
    dim = ctrl.act_mean.numel()
    mu, sigma = torch.zeros(cfg["horizon"], dim, device=goal.device), torch.ones(cfg["horizon"], dim, device=goal.device)
    for _ in range(cfg["iterations"]):
        u = mu + sigma * torch.randn(cfg["samples"], cfg["horizon"], dim, generator=generator).to(goal.device)
        cost = model.get_cost(emb[None], latents[None], goal[None], physical(u, ctrl, cfg)[None], ctrl, cfg["window"])[0]
        elites = u[cost.topk(cfg["elites"], largest=False).indices]
        mu, sigma = elites.mean(0), elites.std(0) + 1e-3
    return physical(u[cost.argmin()] if cfg["pick"] == "best" else mu, ctrl, cfg)
