import torch
import random
import numpy as np
import contextlib
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import math
import json


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_autocast(precision, device):
    """bf16 or fp16 mixed precision, or plain fp32"""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device.type, dtype=dtype)


def git_info():
    """Commit of the code"""
    run = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def make_run_dir(cfg):
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    git = git_info()
    if git["dirty"]:
        diff = subprocess.check_output(["git", "diff", "HEAD"], text=True)
        (run_dir / "diff.patch").write_text(diff)
    yaml.safe_dump(
        cfg | {"git": git}, open(run_dir / "config.yaml", "w"), sort_keys=False
    )
    return run_dir


def forever(loader):
    """restarts the loader at the end of each epoch"""
    while True:
        yield from loader


def warmup_cosine(warmup, total):

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    return f


def peak_memory_gb(device):
    """Peak memory held by tensors on the GPU, None on CPU"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return None


def log(run_dir, step, metrics, wandb_run=None):
    """One line in run_dir/metrics.jsonl, the same metrics to W&B if enabled, a short print"""
    metrics = {k: round(v, 6) for k, v in metrics.items() if v is not None}
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": step} | metrics) + "\n")
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    keys = ["loss", "tf", "kl", "sigreg", "gnorm", "lr_adamw", "clips_per_s"]
    print(
        f"step {step} | "
        + " | ".join(f"{k} {metrics[k]:.4g}" for k in keys if k in metrics)
    )
