import torch
from h2r.utils import seed_everything, make_autocast, make_run_dir
import yaml


def train(cfg):
    seed_everything(cfg["seed"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    torch.set_float32_matmul_precision("high")
    autocast = make_autocast(cfg["precision"], device)

    cfg["model"] = yaml.safe_load(open(cfg["model"]))
    cfg["data"] = yaml.safe_load(open(cfg["data"]))
    run_dir = make_run_dir(cfg)

