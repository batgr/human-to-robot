"""Helpers shared by the world model and controller trainings"""

import json
import math
import random
import socket
import subprocess
import time
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download


def warmup_cosine(warm, total_steps):
    """lr factor: linear warmup over warm steps, then cosine down to 0"""

    def schedule(step):
        if step < warm:
            return step / max(warm, 1)
        return 0.5 * (
            1 + math.cos(math.pi * (step - warm) / max(total_steps - warm, 1))
        )

    return schedule


def git_commit():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def prepare_run(cfg):
    """Full config and run directory: a new run, or the one being resumed (with its own config)"""
    resume = cfg.pop("resume", None)
    if resume:
        run_dir = Path(resume)
        return yaml.safe_load(open(run_dir / "config.yaml")), run_dir, True
    if isinstance(cfg["model"], str):
        cfg["model"] = yaml.safe_load(
            open(cfg["model"])
        )  # the run keeps the full config
    run_dir = Path(cfg["output_dir"]) / cfg["name"] / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    cfg.pop("config_path", None)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    return cfg, run_dir, False


def make_logger(run_dir):
    log_file = open(run_dir / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    return log


def setup(cfg):
    """Seeds, device and mixed precision"""
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    autocast = torch.autocast(
        device, dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16"
    )
    return device, autocast


def download_data(data):
    """Downloads the shards once if they are not there"""
    if not (Path(data["root"]) / "manifest.parquet").exists():
        snapshot_download(data["hf_repo"], repo_type="dataset", local_dir=data["root"])


def new_metadata(cfg, device, sizes, n_params):
    """sizes: number of training and validation samples, e.g. {"train_clips": ..., "val_clips": ...}"""
    hf_repo = cfg["data"]["hf_repo"]
    try:
        data_revision = HfApi().dataset_info(hf_repo).sha
    except Exception:
        data_revision = None
    return (
        {
            "seed": cfg["seed"],
            "git": git_commit(),
            "host": socket.gethostname(),
            "device": device,
            "torch": torch.__version__,
            "hf_repo": hf_repo,
            "data_revision": data_revision,
        }
        | sizes
        | {"trainable_params": n_params}
    )


def resumed_metadata(meta_path):
    meta = json.load(open(meta_path))
    meta["resumed"] = meta.get("resumed", []) + [time.strftime("%Y-%m-%d %H:%M:%S")]
    return meta


def init_wandb(cfg, run_dir, meta, meta_path):
    """W&B run, resumed with the id kept in the metadata; None if disabled"""
    if not cfg["wandb"]["enabled"]:
        return None
    import wandb

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['name']}/{run_dir.name}",
        config=cfg,
        dir=run_dir,
        id=meta.get("wandb_id"),
        resume="allow",
    )
    meta["wandb_id"] = run.id
    json.dump(meta, open(meta_path, "w"), indent=2)
    return wandb


def save_checkpoint(path, model, opts, scheds, epoch, step, best):
    state = {
        k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
    }  # frozen, not saved
    torch.save(
        {
            "model": state,
            "opts": [o.state_dict() for o in opts],
            "scheds": [s.state_dict() for s in scheds],
            "epoch": epoch,
            "step": step,
            "best": best,
        },
        path,
    )


def load_checkpoint(path, model, opts, scheds, device):
    """Restores weights, optimizers and schedules; returns epoch, step, best"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(
        ckpt["model"], strict=False
    )  # the frozen encoder is not saved
    for o, s, so, ss in zip(opts, scheds, ckpt["opts"], ckpt["scheds"]):
        o.load_state_dict(so)
        s.load_state_dict(ss)
    return ckpt["epoch"], ckpt["step"], ckpt["best"]
