"""Pick success in closed loop: the central measure (h2r eval pick).

Each held-out test episode: the simulator is reset to its frame t0 and each policy sends `steps` commands (each held
frameskip control steps) towards the goal frames of the episode. A pick succeeds when the object is above robosuite's
Lift threshold (4 cm) while grasped. The sub-steps explain failures: gripper within 3 cm of the object, object grasped.

Policies (add one: a class with reset(episode) and __call__(obs, state) -> OSC command, in POLICIES):
- random   uniform commands, the floor;
- replay   the demonstration's measured actions through the command map, open loop: checks the action interface (not
           a ceiling: a closed-loop policy can correct what a replay cannot);
- planner  CEM with the controller in the loop, replanning after every observation (MPC, first action sent).
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml

from h2r.data.prepare import states_to_actions
from h2r.data.sim import random_action
from h2r.envs.robosuite_env import RobosuiteEnv
from h2r.eval.episodes import command_map, held_out_episodes, make_env, to_command
from h2r.plan import context_latents, encode, load, plan
from h2r.training.utils import make_logger, prepare_run, setup


class RandomPolicy:
    def __init__(self, cfg, sim_cfg, device):
        self.task = sim_cfg["tasks"][cfg["task"]]
        self.rng = np.random.default_rng(cfg["seed"])

    def reset(self, episode):
        pass

    def __call__(self, obs, state):
        return random_action(self.rng, obs, self.task)


class ReplayPolicy:
    def __init__(self, cfg, sim_cfg, device):
        self.cmap = command_map(cfg, sim_cfg)

    def reset(self, episode):
        self.actions, self.k = states_to_actions(episode["states"])[episode["t0"] :], 0

    def __call__(self, obs, state):
        a = self.actions[self.k] if self.k < len(self.actions) else np.zeros(7)
        self.k += 1
        return to_command(a, state, *self.cmap)


class PlannerPolicy:
    """The context holds the last real frames and the measured actions linking them. With several goal frames
    (subgoals), the steps are shared equally between them."""

    def __init__(self, cfg, sim_cfg, device):
        self.cfg, self.p, self.device = cfg, cfg["planner"], device
        self.model, self.ctrl = load(cfg["controller"], device)
        self.cmap = command_map(cfg, sim_cfg)
        self.gen = torch.Generator().manual_seed(cfg["seed"])
        self.seconds = []  # planning time of every command

    def reset(self, episode):
        t0 = episode["t0"]
        self.goals = encode(self.model, [episode["frames"][g] for g in episode["goals"]], self.device)
        self.per_goal = -(-self.cfg["steps"] // len(self.goals))
        self.imgs, self.states = list(episode["frames"][:t0]), list(episode["states"][:t0])
        self.k = 0

    def __call__(self, obs, state):
        goal = self.goals[min(self.k // self.per_goal, len(self.goals) - 1)]
        self.k += 1
        self.imgs.append(RobosuiteEnv.image(obs, self.cfg["camera"]))
        self.states.append(state)
        start = time.perf_counter()
        context = slice(-self.p["context"], None)
        emb = encode(self.model, self.imgs[context], self.device)
        past = torch.tensor(states_to_actions(np.stack(self.states[context])), dtype=torch.float32, device=self.device)
        latents = context_latents(self.model, self.ctrl, emb, past, self.p["past_latents"])
        action = plan(self.model, self.ctrl, emb, latents, goal, self.p, self.gen)[0].cpu().numpy()
        self.seconds.append(time.perf_counter() - start)
        return to_command(action, state, *self.cmap)


POLICIES = {"random": RandomPolicy, "replay": ReplayPolicy, "planner": PlannerPolicy}


def run_episode(env, policy, episode, steps, frameskip, object_key, camera=None):
    """Closed loop from the frame t0. Returns the episode's outcome, and its frames when a camera is given"""
    obs = env.restore(episode["sim"][episode["t0"]]["sim_state"])
    policy.reset(episode)
    out = {"success": False, "reached": False, "grasped": False, "time_to_success": None}
    frames = [env.image(obs, camera)] if camera else []
    for k in range(steps):
        obs = env.step(policy(obs, env.robot_state(obs)), frameskip)[0]
        grasped = env.grasped()
        out["reached"] |= bool(np.linalg.norm(obs["robot0_eef_pos"] - obs[object_key]) < 0.03)
        out["grasped"] |= grasped
        if grasped and env.success() and not out["success"]:  # lifted while held, not knocked up
            out["success"], out["time_to_success"] = True, k + 1
        if camera:
            frames.append(env.image(obs, camera))
    return out, frames


def wilson(k, n, z=1.96):
    """95% confidence interval of a success rate (Wilson score)"""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z**2 / n
    c, h = (p + z**2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(c - h), float(c + h)


def summarize(rows):
    n, wins = len(rows), sum(r["success"] for r in rows)
    times = [r["time_to_success"] for r in rows if r["success"]]
    return {"episodes": n, "success": wins / n, "success_ci": wilson(wins, n),
            "reached": float(np.mean([r["reached"] for r in rows])),
            "grasped": float(np.mean([r["grasped"] for r in rows])),
            "time_to_success": float(np.mean(times)) if times else None}


def filmstrip(frames, goals, title, path, columns=8):
    """Frames of a closed-loop episode at regular steps, then its goal frames"""
    idx = np.linspace(0, len(frames) - 1, min(columns, len(frames))).round().astype(int)
    panels = [(frames[i], f"step {i}") for i in idx]
    panels += [(g, "goal" if k == len(goals) - 1 else f"subgoal {k + 1}") for k, g in enumerate(goals)]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.6 * len(panels), 2.0))
    for ax, (img, label) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(cfg):
    cfg, run_dir, done = prepare_run(cfg)
    if done:
        print(f"already evaluated: {run_dir}")
        return
    log = make_logger(run_dir)
    device, _ = setup(cfg)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    policies = {name: POLICIES[name](cfg, sim_cfg, device) for name in cfg["policies"]}
    object_key = sim_cfg["tasks"][cfg["task"]]["object"]
    env = make_env(cfg, sim_cfg)
    rows, filmed = [], set()
    for episode in held_out_episodes(cfg, sim_cfg):
        row = {"episode": episode["name"]}
        for name, policy in policies.items():
            film = name == cfg.get("filmstrip") and len(filmed) < 2
            row[name], frames = run_episode(env, policy, episode, cfg["steps"], sim_cfg["frameskip"], object_key,
                                            cfg["camera"] if film else None)
            kind = "success" if row[name]["success"] else "failure"
            if film and kind not in filmed:
                filmed.add(kind)
                filmstrip(frames, [episode["frames"][g] for g in episode["goals"]], f"{name} · {episode['name']} · {kind}",
                          run_dir / f"episode_{kind}.png")
        rows.append(row)
        log(f"{episode['name']} | " + " | ".join(f"{n} {'success' if row[n]['success'] else 'fail'}" for n in policies))
    summary = {n: summarize([r[n] for r in rows]) for n in policies}
    if "planner" in policies:
        seconds = policies["planner"].seconds
        summary["planner"] |= {"seconds_p50": float(np.percentile(seconds, 50)),
                               "seconds_p95": float(np.percentile(seconds, 95))}
    json.dump({"summary": summary, "episodes": rows}, open(run_dir / "results.json", "w"), indent=2)
    for n, m in summary.items():
        lo, hi = m["success_ci"]
        log(f"{n}: success {m['success']:.2f} [{lo:.2f}, {hi:.2f}] on {m['episodes']} episodes | reached "
            f"{m['reached']:.2f} | grasped {m['grasped']:.2f}")
