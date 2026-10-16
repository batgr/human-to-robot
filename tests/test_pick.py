import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from h2r.data.prepare import states_to_actions
from h2r.data.sim import episode
from h2r.eval.episodes import fit_command_map, goal_frames, load_episode, make_env, play, to_command
from h2r.eval.pick import POLICIES, RandomPolicy, ReplayPolicy, run_episode


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


def test_baselines_in_closed_loop(tmp_path):
    """A collected episode, then from its frame t0: the random baseline is reproducible with its seed, and the
    demonstration's actions sent through the command map end near where the demonstration ended"""
    sim_cfg = yaml.safe_load(open("configs/data/sim.yaml"))
    sim_cfg.update(raw_dir=str(tmp_path), steps=10, split={"val": 0, "test": 0, "seed": 0})  # all in train
    sim_cfg["env"].update(camera_heights=64, camera_widths=64)
    folder = episode((sim_cfg, "Panda", "Lift", "scripted", 0))
    cfg = yaml.safe_load(open("configs/eval/pick.yaml"))
    cfg.update(goal="offset", goal_offset=4, steps=8)
    assert set(POLICIES) == {"random", "replay", "planner"}
    ep = load_episode(folder, cfg, sim_cfg)
    env = make_env(cfg, sim_cfg)
    runs = [run_episode(env, RandomPolicy(cfg, sim_cfg, "cpu"), ep, cfg["steps"], sim_cfg["frameskip"], "cube_pos")[0]
            for _ in range(2)]
    assert runs[0] == runs[1]
    out, frames = run_episode(env, ReplayPolicy(cfg, sim_cfg, "cpu"), ep, cfg["steps"], sim_cfg["frameskip"],
                              "cube_pos", cfg["camera"])
    assert set(out) == {"success", "reached", "grasped", "time_to_success"} and len(frames) == cfg["steps"] + 1
    demo = states_to_actions(ep["states"])[ep["t0"] :]
    obs = play(env, ep, demo, fit_command_map([folder]), sim_cfg["frameskip"])[0]
    assert np.linalg.norm(obs["robot0_eef_pos"] - ep["sim"][-1]["eef_world"]) < 0.03
    env.step(np.r_[np.zeros(6), 1.0], 10)  # the gripper target carries over unless restore puts it back
    obs = env.restore(ep["sim"][ep["t0"]]["sim_state"])
    for t in range(ep["t0"], ep["t0"] + 3):
        obs = env.step(np.asarray(ep["sim"][t]["command"]), sim_cfg["frameskip"])[0]
    assert abs(env.robot_state(obs)[6] - ep["states"][ep["t0"] + 3, 6]) < 0.02


def test_lift_goal_frames():
    """goal: lift takes the highest frame if the object rose 4 cm; subgoals add the frames where the gripper starts
    and stops closing"""
    states = np.zeros((20, 7))
    states[8:, 6] = [0.3, 0.6, 0.9] + [1.0] * 9
    height = [0.83] * 12 + [0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93, 0.92]
    sim = [{"object": [0, 0, h]} for h in height]
    cfg = {"start": 2, "goal": "lift", "goal_offset": 10, "subgoals": True}
    assert goal_frames(states, sim, cfg) == [7, 11, 18]
    assert goal_frames(states, sim, cfg | {"subgoals": False}) == [18]
    assert goal_frames(states, [{"object": [0, 0, 0.83]}] * 20, cfg) is None
    assert goal_frames(states, sim, cfg | {"goal": "offset"}) == [12]
