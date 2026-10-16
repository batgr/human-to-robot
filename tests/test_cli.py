from h2r.cli import override
from h2r.training.utils import prepare_run


def test_override_reads_yaml_values_and_nested_keys():
    cfg = override({"name": "h2r", "data": {"max_videos": None}}, ["name=wm-A", "data.max_videos=10",
                                                                   "data.include=[Panda/Lift/*]", "init_from=null"])
    assert cfg == {"name": "wm-A", "data": {"max_videos": 10, "include": ["Panda/Lift/*"]}, "init_from": None}


def test_run_directory_is_reused(tmp_path):
    """A new run writes its config; it is done once it has last.pt (training) or results.json (evaluation), and then
    keeps its own config"""
    cfg, run_dir, done = prepare_run({"name": "wm-A", "output_dir": str(tmp_path), "seed": 0})
    assert run_dir == tmp_path / "wm-A" and not done
    (run_dir / "last.pt").touch()
    cfg, _, done = prepare_run({"name": "wm-A", "output_dir": str(tmp_path), "seed": 1})
    assert done and cfg["seed"] == 0
