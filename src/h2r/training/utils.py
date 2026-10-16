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
    """Run directory output_dir/name and full config. Returns done = True when the run already has its outputs: a
    training continues from last.pt with its own config, an evaluation with results.json is not run again"""
    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    if (run_dir / "last.pt").exists() or (run_dir / "results.json").exists():
        return yaml.safe_load(open(run_dir / "config.yaml")), run_dir, True
    if isinstance(cfg.get("model"), str):
        cfg["model"] = yaml.safe_load(
            open(cfg["model"])
        )  # the run keeps the full config
    run_dir.mkdir(parents=True, exist_ok=True)
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
        device, dtype=torch.bfloat16, enabled=cfg.get("precision") == "bf16"
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


def record(run_dir, wandb, metrics, step):
    """Appends the metrics of a step to run_dir/metrics.jsonl (curves.png), and logs them to W&B when enabled"""
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": step} | metrics) + "\n")
    if wandb:
        wandb.log(metrics, step=step)


def plot_curves(run_dir, smooth=50):
    """Every metric of metrics.jsonl against the step, in run_dir/curves.png. After a resume, a step logged twice keeps
    its last value; train metrics are smoothed over `smooth` steps"""
    import matplotlib.pyplot as plt
    import numpy as np

    curves = {}
    for line in open(run_dir / "metrics.jsonl"):
        row = json.loads(line)
        for k, v in row.items():
            if k != "step":
                curves.setdefault(k, {})[row["step"]] = v
    names = sorted(curves)
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(names) / cols), cols, figsize=(4 * cols, 2.6 * math.ceil(len(names) / cols)),
                             squeeze=False)
    for ax, name in zip(axes.flat, names):
        steps, values = zip(*sorted(curves[name].items()))
        values = np.array(values, dtype=float)
        if name.startswith("train/") and len(values) > smooth:
            values = np.convolve(values, np.ones(smooth) / smooth, mode="valid")
            steps = steps[smooth - 1 :]
        ax.plot(steps, values, lw=1)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    plt.close(fig)


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
