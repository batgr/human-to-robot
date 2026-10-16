"""Does the planner's cost rank action sequences like their real outcome? (h2r eval ranking)

From the frame t0 of each held-out test episode, candidate sequences of `horizon` actions (drawn as the CEM draws them)
are scored by the world model (latent distance to the first goal frame) and played in the simulator from the same
restored state. Real outcome: distance of the gripper and of the object to their positions at the goal frame, so that
reaching the right place without moving the object is not enough. Spearman between cost and outcome, and regret: how
much worse (cm) the sequence the world model prefers is than the best one.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from h2r.eval.episodes import command_map, held_out_episodes, make_env, planner_context, play
from h2r.plan import encode, load, physical
from h2r.training.utils import make_logger, prepare_run, setup


@torch.no_grad()
def run(cfg):
    cfg, run_dir, done = prepare_run(cfg | {"name": f"{cfg['name']}/ranking"})
    if done:
        print(f"already evaluated: {run_dir}")
        return
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    model, ctrl = load(cfg["controller"], device)
    env, cmap, p = make_env(cfg, sim_cfg), command_map(cfg, sim_cfg), cfg["planner"]
    object_key = sim_cfg["tasks"][cfg["task"]]["object"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    rows, shown = [], None
    for ep in held_out_episodes(cfg, sim_cfg):
        obs = env.restore(ep["sim"][ep["t0"]]["sim_state"])
        emb, latents = planner_context(model, ctrl, env, obs, ep, cfg, device)
        g = ep["goals"][0]
        goal = encode(model, ep["frames"][g : g + 1], device)[0]
        target = np.r_[ep["sim"][g]["eef_world"], ep["sim"][g]["object"]]
        u = torch.randn(cfg["ranking"]["candidates"], cfg["horizon"], ctrl.act_mean.numel(), generator=gen)
        acts = physical(u.to(device), ctrl, p)
        cost = model.get_cost(emb[None], latents[None], goal[None], acts[None], ctrl, p["window"])[0].cpu().numpy()
        real = []
        for a in acts.cpu().numpy():
            obs = play(env, ep, a, cmap, sim_cfg["frameskip"])[0]
            reached = np.r_[obs["robot0_eef_pos"], obs[object_key]]
            real.append((np.linalg.norm(reached[:3] - target[:3]) + np.linalg.norm(reached[3:] - target[3:])) * 100)
        real = np.array(real)
        rows.append({"episode": ep["name"], "spearman": float(spearmanr(cost, real).statistic),
                     "regret_cm": float(real[cost.argmin()] - real.min())})
        shown = shown or (cost, real)
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in ("spearman", "regret_cm")}
    json.dump({"summary": summary, "episodes": rows}, open(run_dir / "results.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(*shown, s=10, alpha=0.7)
    ax.set_xlabel("world-model cost")
    ax.set_ylabel("real distance to the goal, gripper + object (cm)")
    ax.set_title(f"candidates of one start ({rows[0]['episode']})", fontsize=9)
    fig.savefig(run_dir / "ranking.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"{len(rows)} starts | Spearman {summary['spearman']:.2f} | regret {summary['regret_cm']:.1f} cm")
