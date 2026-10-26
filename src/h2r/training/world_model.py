import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from h2r.data.dataset import ClipDataset, shards
from h2r.training.utils import (
    seed_everything,
    make_autocast,
    make_run_dir,
    forever,
)
from h2r.models.jepa import build_model
from h2r.models.module import SIGReg


def build_loaders(cfg, device):
    clip_len = cfg["context"] + cfg["rollout_steps"]
    root = cfg["data"]["processed_dir"]
    ds_train = ClipDataset(root, "train", clip_len)
    args = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["num_workers"] > 0,
    )
    train = DataLoader(ds_train, shuffle=True, drop_last=True, **args)
    val = None
    if shards(root, "val"):  # no validation split in the local data
        ds_val = ClipDataset(root, "val", clip_len)
        val = DataLoader(ds_val, shuffle=False, drop_last=False, **args)
    return train, val


def train(cfg):
    seed_everything(cfg["seed"])
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    torch.set_float32_matmul_precision("high")
    autocast = make_autocast(cfg["precision"], device)

    train_loader, val_loader = build_loaders(cfg, device)
    train_iter = forever(train_loader)

    run_dir = make_run_dir(cfg)

    model = build_model(cfg["model"]).to(device)
    sigreg = SIGReg().to(device)


def losses(model, sigreg, emb, cfg):
    C = cfg["context"]
    B = cfg["loss"]["beta"]
    sig_w = cfg["loss"]["sigreg"]
    z, kl = model.idm(emb)

    pred = model.predict(emb[:, :C], z[:, :C])  # (B, C, N, D)

    tf = F.mse_loss(pred.float(), emb[:, 1 : C + 1].float())

    sig = sigreg(rearrange(emb.float(), "b t n d -> (t n) b d"))

    loss = tf + (B * kl) + (sig_w * sig)

    return {"loss": loss, "tf": tf, "sigreg": sig_w * sig,}
