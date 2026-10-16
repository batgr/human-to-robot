"""Does a command produce the displacement it asks for? (h2r eval interface)

Recorded measured actions of the held-out test episodes (one every `every` frames), sent through the command map from
their recorded state: error between requested and obtained displacement per axis (cm), commands clipped at the OSC
limit, gripper command equal to the demonstration's. Checks the interface between the controller's actions and the simulator,
independently of any model.
"""

import json

import numpy as np
import yaml

from h2r.data.prepare import states_to_actions
from h2r.eval.episodes import command_map, held_out_episodes, make_env, to_command
from h2r.training.utils import make_logger, prepare_run


def run(cfg):
    cfg, run_dir, done = prepare_run(cfg | {"name": f"{cfg['name']}/interface"})
    if done:
        print(f"already evaluated: {run_dir}")
        return
    log = make_logger(run_dir)
    sim_cfg = yaml.safe_load(open(cfg["sim_config"]))
    env, cmap = make_env(cfg, sim_cfg), command_map(cfg, sim_cfg)
    err, saturated, gripper, n = np.zeros(3), 0, 0, 0
    for ep in held_out_episodes(cfg, sim_cfg):
        for t in range(0, len(ep["sim"]) - 1, cfg["interface"]["every"]):
            obs = env.restore(ep["sim"][t]["sim_state"])
            before = env.robot_state(obs)
            want = states_to_actions(ep["states"][t : t + 2])[0]
            after = env.robot_state(env.step(to_command(want, before, *cmap), sim_cfg["frameskip"])[0])
            err += np.abs((after - before)[:3] - want[:3]) * 100
            saturated += int((np.abs(cmap[0] * want[:3] + cmap[1]) > 1).any())
            gripper += int(to_command(want, before, *cmap)[6] == ep["sim"][t]["command"][6])
            n += 1
    summary = {"error_cm": (err / n).tolist(), "saturated": saturated / n, "gripper_agrees": gripper / n, "steps": n}
    json.dump(summary, open(run_dir / "results.json", "w"), indent=2)
    log("error per axis " + " ".join(f"{e:.2f}" for e in summary["error_cm"]) + f" cm | saturated "
        f"{summary['saturated']:.2f} | gripper agrees {summary['gripper_agrees']:.2f} ({n} commands)")
