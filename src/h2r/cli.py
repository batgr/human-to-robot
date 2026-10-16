"""Command line: h2r <group> <command> [--config path], h2r train | h2r controller train [--config path] [--resume run_dir]"""

import argparse

import yaml

from h2r.data import prepare, sim


def train(cfg):
    from h2r.training.world_model import train as run  # torch and transformers load only when training

    run(cfg)


def train_controller(cfg):
    from h2r.training.controller import train_controller as run

    run(cfg)


EVALS = {  # h2r eval <name>: module h2r.eval.<name>, default config, help
    "pick": ("configs/eval/pick.yaml", "pick success in closed loop, the central measure"),
    "transfer": ("configs/eval/transfer.yaml", "human-to-robot transfer from pick results (arms A, B, C)"),
    "prediction": ("configs/eval/diagnostics.yaml", "skill(h) of the world model on robot episodes"),
    "counterfactual": ("configs/eval/diagnostics.yaml", "different actions, different and coherent predictions?"),
    "ranking": ("configs/eval/diagnostics.yaml", "planner cost vs real outcome of played sequences"),
    "interface": ("configs/eval/diagnostics.yaml", "requested vs obtained displacement of the commands"),
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


def main():
    parser = argparse.ArgumentParser(prog="h2r")
    groups = parser.add_subparsers(dest="group", required=True)
    for group, commands in COMMANDS.items():
        sub = groups.add_parser(group).add_subparsers(dest="command", required=True)
        for name, (fn, help, config) in commands.items():
            cmd = sub.add_parser(name, help=help)
            cmd.add_argument("--config", default=config)
            cmd.set_defaults(fn=fn)

    cmd = groups.add_parser("train", help="train the world model and the IDM")
    cmd.add_argument("--config", default="configs/training/world_model.yaml")
    cmd.add_argument("--resume", help="run directory to continue")
    cmd.set_defaults(fn=train)

    sub = groups.add_parser("controller", help="controller for a trained world model").add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("train", help="train a controller for a trained world model")
    cmd.add_argument("--config", default="configs/training/controller.yaml")
    cmd.add_argument("--resume", help="run directory to continue")
    cmd.set_defaults(fn=train_controller)

    sub = groups.add_parser("eval", help="evaluations on robosuite episodes").add_subparsers(dest="command", required=True)
    for name, (config, help) in EVALS.items():
        cmd = sub.add_parser(name, help=help)
        cmd.add_argument("--config", default=config)
        cmd.set_defaults(fn=evaluation(name))

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config)) | {"config_path": args.config}
    if getattr(args, "resume", None):
        cfg["resume"] = args.resume
    args.fn(cfg)
