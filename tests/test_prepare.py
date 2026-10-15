import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from h2r.data.prepare import extract, frame_actions, states_to_actions


def vjepa2_actions(poses):
    """poses_to_diffs of V-JEPA 2-AC (app/vjepa_droid/droid.py), the reference definition"""
    matrices = [Rotation.from_euler("xyz", t).as_matrix() for t in poses[:, 3:6]]
    turn = [Rotation.from_matrix(matrices[t + 1] @ matrices[t].T).as_euler("xyz") for t in range(len(poses) - 1)]
    return np.concatenate([np.diff(poses[:, :3], axis=0), np.stack(turn), np.diff(poses[:, 6:], axis=0)], axis=1)


def test_actions_match_vjepa2():
    poses = np.random.default_rng(0).uniform(-3, 3, (20, 7))
    assert np.allclose(states_to_actions(poses), vjepa2_actions(poses), atol=1e-9)


def test_rotation_does_not_jump_at_pi():
    poses = np.zeros((2, 7))
    poses[:, 5] = [3.1, -3.1]  # yaw crosses pi: a small turn, not -6.2 rad
    assert abs(states_to_actions(poses)[0, 5] - (2 * np.pi - 6.2)) < 1e-9


def test_states_sampled_at_kept_frames(tmp_path):
    times = np.arange(30) / 15  # 15 fps
    states = np.zeros((30, 7))
    states[:, 0] = times  # x grows by 1 m/s
    pq.write_table(pa.table({"timestamp": times, "state": states.tolist()}), tmp_path / "state.parquet")
    actions = np.asarray(frame_actions(tmp_path / "state.parquet", 5, fps=4))  # frames at 0, 0.25, ... 1 s
    nearest = np.array([0, 4, 7, 11, 15]) / 15  # nearest 15 fps timestamps (0.5 s: tie, the earlier one)
    assert np.allclose(actions[:, 0], np.diff(nearest), atol=1e-6)


def video(path, seconds=2, fps=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate={fps}",
           "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


@pytest.mark.parametrize("with_actions", [True, False])
def test_extract(tmp_path, with_actions):
    raw, out = tmp_path / "raw", tmp_path / "out"
    video(raw / "human/clip.mp4")
    for cam in ("exterior_1_left", "exterior_2_left"):
        video(raw / f"robot/episode_000001/{cam}.mp4")
    times = np.arange(30) / 15
    states = np.random.default_rng(0).normal(size=(30, 7))
    pq.write_table(pa.table({"timestamp": times, "state": states.tolist()}), raw / "robot/episode_000001/state.parquet")
    config = tmp_path / "config.yaml"
    config.write_text("")
    cfg = {"raw_dir": raw, "processed_dir": out, "fps": 4, "image_size": 64, "min_frames": 5, "shard_size": 10,
           "split": {"val": 0.0, "test": 0.0, "seed": 0}, "actions": with_actions, "split_by": "folder",
           "config_path": config}
    extract(cfg)
    rows = {r["video"]: r for p in (out / "frames").glob("*.parquet") for r in pq.read_table(p).to_pylist()}
    assert len(rows) == 3
    if not with_actions:
        assert all("actions" not in r for r in rows.values())  # same shards as before the option
        return
    assert rows["human/clip.mp4"]["actions"] is None
    cams = [rows[f"robot/episode_000001/{c}.mp4"] for c in ("exterior_1_left", "exterior_2_left")]
    assert len(cams[0]["actions"]) == len(cams[0]["frames"]) - 1 == 7  # 2 s at 4 fps: 8 frames
    assert cams[0]["actions"] == cams[1]["actions"]
