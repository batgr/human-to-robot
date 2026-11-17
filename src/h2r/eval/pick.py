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
        self.actions = states_to_actions(episode["states"])[episode["t0"] :]
        self.k = 0

    def __call__(self, obs, state):
        a = self.actions[self.k] if self.k < len(self.actions) else np.zeros(7)
        self.k += 1
        return to_command(a, state, *self.cmap)


class PlannerPolicy:
    """Closed-loop MPC: condition on the last real frames and measured actions.

    Split the control budget evenly between the supplied visual subgoals.
    """

    def __init__(self, cfg, sim_cfg, device):
        self.cfg = cfg
        self.planner_cfg = cfg["planner"]
        self.device = device
        self.model, self.ctrl = load(cfg["controller"], device)
        self.cmap = command_map(cfg, sim_cfg)
        self.gen = torch.Generator().manual_seed(cfg["seed"])
        self.seconds = []

    def reset(self, episode):
        t0 = episode["t0"]
        goal_frames = [episode["frames"][i] for i in episode["goals"]]
        self.goals = encode(self.model, goal_frames, self.device)
        self.per_goal = (self.cfg["steps"] + len(self.goals) - 1) // len(self.goals)

        # Frames before t0 seed the planner's first context window.
        self.imgs = list(episode["frames"][:t0])
        self.states = list(episode["states"][:t0])
        self.k = 0

    def __call__(self, obs, state):
        goal_idx = min(self.k // self.per_goal, len(self.goals) - 1)
        goal = self.goals[goal_idx]
        self.k += 1

        # Rebuild the context from observations, not the previous CEM plan.
        self.imgs.append(RobosuiteEnv.image(obs, self.cfg["camera"]))
        self.states.append(state)
        context = slice(-self.planner_cfg["context"], None)

        start = time.perf_counter()
        emb = encode(self.model, self.imgs[context], self.device)
        measured_actions = states_to_actions(np.stack(self.states[context]))
        past_actions = torch.tensor(measured_actions, dtype=torch.float32, device=self.device)
        latents = context_latents(
            self.model, self.ctrl, emb, past_actions, self.planner_cfg["past_latents"]
        )
        planned_actions = plan(
            self.model, self.ctrl, emb, latents, goal, self.planner_cfg, self.gen
        )
        self.seconds.append(time.perf_counter() - start)

        # Execute only the first action; the next call observes the actual result.
        return to_command(planned_actions[0].cpu().numpy(), state, *self.cmap)

POLICIES = {"random": RandomPolicy, "replay": ReplayPolicy, "planner": PlannerPolicy}


def run_episode(env, policy, episode, steps, frameskip, object_key, camera=None):
    """Closed loop from the frame t0. Returns the episode's outcome, and its frames when a camera is given"""
    obs = env.restore(episode["sim"][episode["t0"]]["sim_state"])
    policy.reset(episode)
    outcome = {"success": False, "reached": False, "grasped": False, "time_to_success": None}
    frames = [env.image(obs, camera)] if camera else []

    for step in range(steps):
        command = policy(obs, env.robot_state(obs))
        obs = env.step(command, frameskip)[0]
        grasped = env.grasped()

        outcome["reached"] |= bool(np.linalg.norm(obs["robot0_eef_pos"] - obs[object_key]) < 0.03)
        outcome["grasped"] |= grasped
        if grasped and env.success() and not outcome["success"]:
            outcome["success"] = True
            outcome["time_to_success"] = step + 1
        if camera:
            frames.append(env.image(obs, camera))

    return outcome, frames


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
    return {
        "episodes": n,
        "success": wins / n,
        "success_ci": wilson(wins, n),
        "reached": float(np.mean([row["reached"] for row in rows])),
        "grasped": float(np.mean([row["grasped"] for row in rows])),
        "time_to_success": float(np.mean(times)) if times else None,
    }


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
    with open(cfg["sim_config"]) as config_file:
        sim_cfg = yaml.safe_load(config_file)
    policies = {name: POLICIES[name](cfg, sim_cfg, device) for name in cfg["policies"]}
    object_key = sim_cfg["tasks"][cfg["task"]]["object"]
    env = make_env(cfg, sim_cfg)
    rows, filmed = [], set()
    for episode in held_out_episodes(cfg, sim_cfg):
        row = {"episode": episode["name"]}
        for name, policy in policies.items():
            capture_frames = name == cfg.get("filmstrip") and len(filmed) < 2
            row[name], frames = run_episode(
                env, policy, episode, cfg["steps"], sim_cfg["frameskip"], object_key,
                cfg["camera"] if capture_frames else None,
            )
            outcome = "success" if row[name]["success"] else "failure"
            if capture_frames and outcome not in filmed:
                filmed.add(outcome)
                goals = [episode["frames"][i] for i in episode["goals"]]
                filmstrip(
                    frames, goals, f"{name} · {episode['name']} · {outcome}",
                    run_dir / f"episode_{outcome}.png",
                )
        rows.append(row)
        log(f"{episode['name']} | " + " | ".join(f"{n} {'success' if row[n]['success'] else 'fail'}" for n in policies))
    summary = {n: summarize([r[n] for r in rows]) for n in policies}
    if "planner" in policies:
        seconds = policies["planner"].seconds
        summary["planner"] |= {"seconds_p50": float(np.percentile(seconds, 50)),
                               "seconds_p95": float(np.percentile(seconds, 95))}
    with open(run_dir / "results.json", "w") as result_file:
        json.dump({"summary": summary, "episodes": rows}, result_file, indent=2)
    for n, m in summary.items():
        lo, hi = m["success_ci"]
        log(f"{n}: success {m['success']:.2f} [{lo:.2f}, {hi:.2f}] on {m['episodes']} episodes | reached "
            f"{m['reached']:.2f} | grasped {m['grasped']:.2f}")
