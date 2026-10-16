"""Does the planner find the actions that lead to a goal a few steps away? (h2r eval planning)

Offline, as Garrido et al. 2026 (protocol of Terver et al. 2025): random clips of the held-out test episodes, the
real context up to frame t and the real frame t + H as goal. The planner searches H actions (CEM, as in h2r eval pick);
Δxyz = L1 distance between the summed planned and the summed recorded displacements (cm), since many paths lead to
the same point. Baseline: no motion, whose Δxyz is the recorded displacement itself.
"""

import json

import numpy as np
import torch
import yaml

from h2r.data.prepare import states_to_actions
from h2r.eval.episodes import held_out_episodes
from h2r.plan import context_latents, encode, load, plan
from h2r.training.utils import make_logger, prepare_run, setup


def displacement_error(planned, recorded):
    """(H, act_dim) actions in meters -> L1 distance of their summed translations, in cm"""
    return float((planned[:, :3].sum(0) - recorded[:, :3].sum(0)).abs().sum() * 100)


def run(cfg):
    cfg, run_dir, done = prepare_run(cfg | {"name": f"{cfg['name']}/planning"})
    if done:
        print(f"already evaluated: {run_dir}")
        return
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    model, ctrl = load(cfg["controller"], device)
    p = cfg["planner"]
    C, H = p["context"], p["horizon"]
    rng, gen = np.random.default_rng(cfg["seed"]), torch.Generator().manual_seed(cfg["seed"])
    episodes = held_out_episodes(cfg, sim_cfg)
    for ep in episodes:
        ep["emb"] = encode(model, ep["frames"], device)
        ep["actions"] = torch.tensor(states_to_actions(ep["states"]), dtype=torch.float32, device=device)
    planned, still = [], []
    for _ in range(cfg["planning"]["clips"]):
        ep = episodes[rng.integers(len(episodes))]
        t = int(rng.integers(C - 1, len(ep["emb"]) - H))
        emb, actions = ep["emb"][t - C + 1 : t + 1], ep["actions"]
        latents = context_latents(model, ctrl, emb, actions[t - C + 1 : t], p["past_latents"])
        recorded = actions[t : t + H]
        planned.append(displacement_error(plan(model, ctrl, emb, latents, ep["emb"][t + H], p, gen), recorded))
        still.append(displacement_error(torch.zeros_like(recorded), recorded))
    planned, still = np.array(planned), np.array(still)
    results = {"dxyz_cm": planned.mean(), "no_motion_dxyz_cm": still.mean(),
               "better_than_no_motion": (planned < still).mean(), "clips": len(planned), "horizon": H}
    json.dump({k: float(v) for k, v in results.items()}, open(run_dir / "results.json", "w"), indent=2)
    log(f"Δxyz {planned.mean():.2f} cm | no motion {still.mean():.2f} cm | better than no motion "
        f"{(planned < still).mean():.2f} | {len(planned)} clips, H = {H}")
