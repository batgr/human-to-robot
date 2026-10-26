import torch
from h2r.utils import seed_everything, make_autocast, make_run_dir
from pathlib import Path


def train(cfg):
    seed_everything(cfg["seed"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    torch.set_float32_matmul_precision("high")
    autocast = make_autocast(cfg["precision"], device)

    run_dir = make_run_dir(cfg)


def build_loaders(cfg, device):
    clip_len = cfg["context"] + cfg["rollout_steps"]
    root = cfg["data"]["processed_dir"]
