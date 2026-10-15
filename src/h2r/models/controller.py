"""Controller from real to latent actions, after Garrido et al. 2026 (arXiv:2601.05230)"""

import torch
from torch import nn

from h2r.models.module import CrossBlock, Transformer


class Controller(nn.Module):
    """Latent action of a real action taken from the current frame, to drive the world model without the IDM"""

    def __init__(
        self,
        *,
        num_patches,
        dim,
        latent_dim=None,
        act_dim=7,
        depth=2,
        heads=4,
        dim_head=None,
        mlp_dim=None,
    ):
        super().__init__()
        dim_head = dim_head or dim // heads
        mlp_dim = mlp_dim or 4 * dim
        latent_dim = latent_dim or dim  # the IDM's latent actions live in the latent dim

        # standardization of the real actions, set from the training actions and saved with the weights
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # position in the image
        self.transformer = Transformer(dim, dim, dim, depth, heads, dim_head, mlp_dim)
        self.act_mlp = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.readout = CrossBlock(dim, heads, dim_head, mlp_dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, latent_dim)

        nn.init.zeros_(self.head.weight)  # start at the prior mean: z_hat = 0
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def set_action_stats(self, actions):
        """actions: (..., act_dim) training actions, in the dataset's units"""
        a = actions.reshape(-1, actions.size(-1)).float()
        std = a.std(0)
        self.act_mean.copy_(a.mean(0))
        self.act_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))  # constant dims stay unscaled

    def forward(self, action, emb):
        """
        action: (..., act_dim) real actions, in the dataset's units
        emb: (..., N, D) representation of the frame the action starts from
        returns z_hat: (..., latent_dim)
        """
        lead, n = emb.shape[:-2], emb.size(-2)
        x = emb.reshape(-1, n, emb.size(-1)) + self.pos_embedding
        x = self.transformer(x, causal=False, frame_size=n)  # one frame: every patch sees every patch
        a = (action.reshape(-1, action.size(-1)) - self.act_mean) / self.act_std
        query = self.act_mlp(a)[:, None]  # the action reads the state
        h = self.readout(query, x, frame_size=n)[:, 0]
        return self.head(self.norm(h)).reshape(*lead, -1)


def build_controller(cfg, wm_cfg):
    """The controller's dimensions come from the world model, the rest from its own config"""
    idm = wm_cfg["model"]["idm"]
    return Controller(
        num_patches=idm["num_patches"],
        dim=idm["dim"],
        latent_dim=idm.get("action_dim"),
        **cfg,
    )
