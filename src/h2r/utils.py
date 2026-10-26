import torch
import random
import numpy as np
import contextlib
import subprocess
from datetime import datetime
from pathlib import Path
import yaml


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_autocast(precision, device):
    """bf16 or fp16 mixed precision, or plain fp32"""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(torch.device(device).type, dtype=dtype)


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
