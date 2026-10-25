import importlib

from h2r.cli import GROUPS, override


def test_override_reads_yaml_values_and_nested_keys():
    cfg = override({"name": "h2r", "data": {"max_videos": None}}, ["name=wm-A", "data.max_videos=10",
                                                                   "data.include=[Panda/Lift/*]", "init_from=null"])
    assert cfg == {"name": "wm-A", "data": {"max_videos": 10, "include": ["Panda/Lift/*"]}, "init_from": None}


def test_every_command_has_its_function_and_config():
    from pathlib import Path

    for target, config, _ in [spec for commands in GROUPS.values() for spec in commands.values()]:
        module, fn = target.split(":")
        assert callable(getattr(importlib.import_module(module), fn)) and Path(config).exists()
