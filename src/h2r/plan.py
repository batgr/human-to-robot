"""Plan with a trained controller: CEM over real actions, scored by JEPA.get_cost, and the offline evaluation.

h2r plan offline: on clips of test episodes, plan from t towards the frame t + H and compare the summed planned
translation with the real one (Δxyz, Garrido et al. 2026, appendix A). The closed loop in robosuite is in
h2r.policy (PlannerPolicy).

The context holds the last real frames and the latent actions linking them: the controller on the executed (measured)
actions, as for the future, or the IDM on the real pairs (past_latents: idm).
"""

import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from h2r.data.dataset import read_episodes
from h2r.models.controller import build_controller
from h2r.models.jepa import load_world_model
from h2r.training.utils import make_logger, prepare_run, setup


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


@torch.no_grad()
def plan(model, ctrl, emb, latents, goal, cfg, generator=None):
    """CEM in the controller's standardized action space (mean 0, std 1 at the start), searched dimensions only.
    Returns (H, act_dim) actions in physical units."""
    dim = ctrl.act_mean.numel()
    searched = torch.zeros(dim, dtype=torch.bool, device=goal.device)
    searched[cfg["searched"]] = True
    mu, sigma = torch.zeros(cfg["horizon"], dim, device=goal.device), torch.ones(cfg["horizon"], dim, device=goal.device)

    def physical(u):
        a = (u * ctrl.act_std + ctrl.act_mean) * searched  # rotations etc. stay at 0
        if cfg.get("max_norm"):
            a[..., :3] *= (cfg["max_norm"] / a[..., :3].norm(dim=-1, keepdim=True)).clamp(max=1)
        return a

    for _ in range(cfg["iterations"]):
        u = mu + sigma * torch.randn(cfg["samples"], cfg["horizon"], dim, generator=generator).to(goal.device)
        cost = model.get_cost(emb[None], latents[None], goal[None], physical(u)[None], ctrl, cfg["window"])[0]
        elites = u[cost.topk(cfg["elites"], largest=False).indices]
        mu, sigma = elites.mean(0), elites.std(0) + 1e-3
    return physical(u[cost.argmin()] if cfg["pick"] == "best" else mu)


# ---------------------------------------------------------------- offline (Δxyz)


def delta_xyz(planned, real):
    """L1 distance between the summed planned and real translations"""
    return float(np.abs(np.asarray(planned)[:, :3].sum(0) - np.asarray(real)[:, :3].sum(0)).sum())


def offline(cfg):
    cfg, run_dir, _ = prepare_run(cfg)
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    model, ctrl = load(cfg["controller"], device)
    p, gen = cfg["planner"], torch.Generator().manual_seed(cfg["seed"])
    episodes = read_episodes(cfg["data"]["root"], cfg["data"]["split"])
    starts = [(e, t) for e, ep in enumerate(episodes) for t in range(p["context"] - 1, len(ep["frames"]) - p["horizon"])]
    results = []
    for e, t in random.Random(cfg["seed"]).sample(starts, min(cfg["clips"], len(starts))):
        ep = episodes[e]
        idx = list(range(t - p["context"] + 1, t + 1)) + [t + p["horizon"]]
        emb = encode(model, [np.asarray(Image.open(io.BytesIO(ep["frames"][i]))) for i in idx], device)
        past = torch.tensor(ep["actions"][idx[0] : t], dtype=torch.float32, device=device)
        latents = context_latents(model, ctrl, emb[:-1], past, p["past_latents"])
        actions = plan(model, ctrl, emb[:-1], latents, emb[-1], p, gen).cpu()
        real = ep["actions"][t : t + p["horizon"]]
        results.append({"video": ep["video"], "t": t, "delta_xyz": delta_xyz(actions, real),
                        "no_motion": delta_xyz(np.zeros_like(real), real)})
    summary = {k: float(np.mean([r[k] for r in results])) for k in ("delta_xyz", "no_motion")}
    json.dump({"summary": summary, "clips": results}, open(run_dir / "results.json", "w"), indent=2)
    log(f"{len(results)} clips | Δxyz {summary['delta_xyz']:.4f} (no motion {summary['no_motion']:.4f})")
