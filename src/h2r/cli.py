"""Command line: h2r <group> <command> [--config path] [--set key=value ...]

Runs go to output_dir/name. Running a training again continues it from last.pt; an evaluation that already wrote
results.json is not run again."""

import argparse

import yaml

from h2r.data import prepare, sim


def train(cfg):
    from h2r.training.world_model import (
        train as run,
    )  # torch and transformers load only when training

    run(cfg)


def train_controller(cfg):
    from h2r.training.controller import train_controller as run

    run(cfg)


EVALS = {  # h2r eval <name>: module h2r.eval.<name>, default config, help
    "pick": (
        "configs/eval/pick.yaml",
        "pick success in closed loop, the central measure",
    ),
    "transfer": (
        "configs/eval/transfer.yaml",
        "human-to-robot transfer from pick results (arms A, B, C)",
    ),
    "prediction": (
        "configs/eval/diagnostics.yaml",
        "skill(h) of the world model on robot episodes",
    ),
    "counterfactual": (
        "configs/eval/diagnostics.yaml",
        "different actions, different and coherent predictions?",
    ),
    "ranking": (
        "configs/eval/diagnostics.yaml",
        "planner cost vs real outcome of played sequences",
    ),
    "interface": (
        "configs/eval/diagnostics.yaml",
        "requested vs obtained displacement of the commands",
    ),
}


def evaluation(name):
    def run(cfg):
        import importlib  # torch and robosuite load only when evaluating

        importlib.import_module(f"h2r.eval.{name}").run(cfg)

    return run


COMMANDS = {
    "data": {
        "collect": (
            sim.collect,
            "robosuite episodes -> raw videos, states and simulator data",
            "configs/data/sim.yaml",
        ),
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


def override(cfg, pairs):
    """key=value pairs, nested keys with dots (data.max_videos=10), values read as YAML ([a, b], null, 0.1)"""
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

    def command(sub, name, fn, config, help):
        cmd = sub.add_parser(name, help=help)
        cmd.add_argument("--config", default=config)
        cmd.add_argument(
            "--set",
            nargs="+",
            action="extend",
            default=[],
            metavar="KEY=VALUE",
            help="override config values, e.g. --set name=wm-A data.max_videos=10",
        )
        cmd.set_defaults(fn=fn)

    for group, commands in COMMANDS.items():
        sub = groups.add_parser(group).add_subparsers(dest="command", required=True)
        for name, (fn, help, config) in commands.items():
            command(sub, name, fn, config, help)
    command(
        groups,
        "train",
        train,
        "configs/training/world_model.yaml",
        "train the world model and the IDM",
    )
    sub = groups.add_parser(
        "controller", help="controller for a trained world model"
    ).add_subparsers(dest="command", required=True)
    command(
        sub,
        "train",
        train_controller,
        "configs/training/controller.yaml",
        "train a controller",
    )
    sub = groups.add_parser(
        "eval", help="evaluations on robosuite episodes"
    ).add_subparsers(dest="command", required=True)
    for name, (config, help) in EVALS.items():
        command(sub, name, evaluation(name), config, help)

    args = parser.parse_args()
    args.fn(
        override(yaml.safe_load(open(args.config)), args.set)
        | {"config_path": args.config}
    )
