"""Command line"""

import argparse
import importlib

import yaml

GROUPS = {
    "data": {
        "extract": (
            "h2r.data.prepare:extract",
            "configs/data/config.yaml",
            "raw videos -> frame shards",
        ),
        "push": (
            "h2r.data.prepare:push",
            "configs/data/config.yaml",
            "frame shards -> Hugging Face Hub",
        ),
    },
    "train": {
        "wm": (
            "h2r.training.world_model:train",
            "configs/training/world_model.yaml",
            "",
        )
    },
}


def load(path):
    """YAML config, every value ending in .yaml replaced by the content of that file"""
    cfg = yaml.safe_load(open(path))
    return {
        k: load(v) if isinstance(v, str) and v.endswith(".yaml") else v
        for k, v in cfg.items()
    }


def override(cfg, pairs):
    """key=value pairs"""
    for pair in pairs:
        key, value = pair.split("=", 1)
        *path, last = key.split(".")
        d = cfg
        for k in path:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(value)
    return cfg


def main():
    parser = argparse.ArgumentParser(prog="h2r")
    groups = parser.add_subparsers(dest="group", required=True)
    for group, commands in GROUPS.items():
        sub = groups.add_parser(group).add_subparsers(dest="command", required=True)
        for name, (target, config, help) in commands.items():
            cmd = sub.add_parser(name, help=help)
            cmd.add_argument("--config", default=config)
            cmd.add_argument(
                "--set",
                nargs="+",
                action="extend",
                default=[],
                metavar="KEY=VALUE",
                help="override config values, e.g. --set raw_dir=data/raw fps=4",
            )
            cmd.set_defaults(target=target)

    args = parser.parse_args()
    module, fn = args.target.split(":")
    cfg = override(load(args.config), args.set) | {"config_path": args.config}
    getattr(importlib.import_module(module), fn)(cfg)
