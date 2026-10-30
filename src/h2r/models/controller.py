"""Controller from real to latent actions, after Garrido et al. 2026 (arXiv:2601.05230)"""

from pathlib import Path

import torch
import yaml
from torch import nn

from h2r.models.jepa import load_world_model


class Controller(nn.Module):
    """Latent action of a real action taken from the current frame, to drive the world model without the IDM:
    an MLP on the standardized action and the state"""

    def __init__(self, *, dim, z_dim=None, act_dim=7, depth=2, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or 4 * dim
        z_dim = z_dim or dim

        # standardization of the real actions, set from the training actions and saved with the weights
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))

        layers, width = [], act_dim + dim
        for _ in range(depth):
            layers += [nn.Linear(width, mlp_dim), nn.SiLU()]
            width = mlp_dim
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(mlp_dim)
        self.head = nn.Linear(mlp_dim, z_dim)  # the IDM's latent actions

        nn.init.zeros_(self.head.weight)  # start at the prior mean: z_hat = 0
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def set_action_stats(self, actions):
        """actions: (..., act_dim) training actions, in the dataset's units"""
        a = actions.reshape(-1, actions.size(-1)).float()
        std = a.std(0)
        self.act_mean.copy_(a.mean(0))
        self.act_std.copy_(
            torch.where(std > 1e-6, std, torch.ones_like(std))
        )  # constant dims stay unscaled

    def forward(self, action, emb):
        """
        action: (..., act_dim) real actions, in the dataset's units
        emb: (..., D) state of the frame the action starts from
        returns z_hat: (..., z_dim)
        """
        a = (action - self.act_mean) / self.act_std
        return self.head(self.norm(self.net(torch.cat([a, emb], dim=-1))))


def build_controller(cfg, wm_cfg):
    """The controller's dimensions come from the world model, the rest from its own config"""
    idm = wm_cfg["model"]["idm"]
    return Controller(dim=idm["dim"], z_dim=idm.get("z_dim"), **cfg)


def load_controller(run_dir, device):
    """Frozen controller of a training run (config.yaml + best.pt) and the frozen world model it was trained for"""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model, wm_cfg = load_world_model(cfg["world_model"], device)
    ctrl = build_controller(cfg["model"], wm_cfg).to(device)
    ctrl.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    return model, ctrl.eval().requires_grad_(False)
