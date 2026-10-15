import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from h2r.data.sim import episode
from h2r.envs.robosuite_env import RobosuiteEnv
from h2r.policy import POLICIES, RandomPolicy, fit_command_map, load_episode, run_episode, to_command


def test_command_map(tmp_path):
    """Fitted on commands vs measured translations, then a planned delta maps back to the command"""
    rng = np.random.default_rng(0)
    commands = rng.uniform(-1, 1, (20, 3))
    pos = np.concatenate([[[0, 0, 0]], np.cumsum(0.03 * commands, 0)])  # measured = 0.03 * command
    states = np.concatenate([pos, np.zeros((21, 4))], 1)
    pq.write_table(pa.table({"timestamp": np.arange(21) / 4, "state": states.tolist()}), tmp_path / "state.parquet")
    sim = [{"command": c.tolist() + [0, 0, 0, 1]} for c in commands] + [{"command": None}]
    pq.write_table(pa.Table.from_pylist(sim), tmp_path / "sim.parquet")
    gain, offset = fit_command_map([tmp_path])
    assert np.allclose(gain, 1 / 0.03, rtol=1e-3) and np.allclose(offset, 0, atol=1e-6)
    command = to_command(np.array([0.015, -0.03, 0.0, 0, 0, 0, 0.8]), np.zeros(7), gain, offset)
    assert np.allclose(command, [0.5, -1.0, 0, 0, 0, 0, 1.0], atol=1e-3)  # closes: planned closedness 0.8 > 0.5


def test_random_policy_in_closed_loop(tmp_path):
    """A collected episode, then the random baseline from its frame t0: reproducible with its seed"""
    sim_cfg = yaml.safe_load(open("configs/data/sim.yaml"))
    sim_cfg.update(raw_dir=str(tmp_path), steps=8)
    sim_cfg["env"].update(camera_heights=64, camera_widths=64)
    folder = episode((sim_cfg, "Lift", "scripted", 0))
    cfg = yaml.safe_load(open("configs/policy/robosuite.yaml"))
    cfg.update(start=2, goal_offset=4, steps=3)
    assert set(POLICIES) == {"random", "planner"}
    ep = load_episode(folder, cfg, sim_cfg)
    env = RobosuiteEnv({"env_name": "Lift", "camera_names": [cfg["camera"]], **sim_cfg["env"]})
    env.reset()
    runs = [run_episode(env, RandomPolicy(cfg, sim_cfg, "cpu"), ep, cfg["steps"], sim_cfg["frameskip"]) for _ in range(2)]
    assert np.allclose(runs[0], runs[1], atol=1e-3) and all(0 < d < 1 for d in runs[0])  # same seed: within 1 mm
