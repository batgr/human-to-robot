"""Command line: h2r <group> <command> [--config path]"""

import argparse

import yaml

from h2r.data import prepare

COMMANDS = {
    "data": {
        "extract": (
            prepare.extract,
            "raw videos -> frame shards",
            "configs/data/config.yaml",
        ),
        "push": (
            prepare.push,
            "frame shards -> Hugging Face Hub",
            "configs/data/config.yaml",
        ),
    },
}


def main():
    parser = argparse.ArgumentParser(prog="h2r")
    groups = parser.add_subparsers(dest="group", required=True)
    for group, commands in COMMANDS.items():
        sub = groups.add_parser(group).add_subparsers(dest="command", required=True)
        for name, (fn, help, config) in commands.items():
            cmd = sub.add_parser(name, help=help)
            cmd.add_argument("--config", default=config)
            cmd.set_defaults(fn=fn)

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config)) | {"config_path": args.config}
    args.fn(cfg)
