import subprocess

import numpy as np
import pyarrow.parquet as pq
import yaml

from h2r.data.sim import episode


def n_frames(path):
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
           "stream=nb_read_frames", "-of", "csv=p=0", str(path)]
    return int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def test_episode_folder(tmp_path):
    cfg = yaml.safe_load(open("configs/data/sim.yaml"))
    cfg.update(raw_dir=str(tmp_path), steps=4, cameras=["agentview", "frontview"])
    cfg["env"].update(camera_heights=64, camera_widths=64)
    out = episode((cfg, "Lift", "scripted", 0))
    assert out == tmp_path / "Lift" / "scripted" / "episode_000000"
    for cam in cfg["cameras"]:
        assert n_frames(out / f"{cam}.mp4") == 5  # steps + 1 frames, one per action
    state = pq.read_table(out / "state.parquet")
    assert np.allclose(state["timestamp"].to_pylist(), np.arange(5) / 4)  # 20 Hz / frameskip 5
    states = np.asarray(state["state"].to_pylist())
    assert states.shape == (5, 7) and ((states[:, 6] >= 0) & (states[:, 6] <= 1)).all()
    sim = pq.read_table(out / "sim.parquet").to_pylist()
    assert [s["command"] is None for s in sim] == [False] * 4 + [True]  # no action after the last frame
    assert len(sim[0]["sim_state"]) > 0 and len(sim[0]["object"]) == 3
