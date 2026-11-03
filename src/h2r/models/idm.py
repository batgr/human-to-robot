"""Inverse dynamics model, after Garrido et al. 2026 (arXiv:2601.05230)"""

import torch
from einops import rearrange
from torch import nn

from h2r.models.module import CrossBlock, Transformer


class IDM(nn.Module):
    """Latent action from two consecutive frames, regularized like a VAE"""

    def __init__(
        self,
        *,
        dim,
        z_dim=None,
        depth=2,
        heads=4,
        dim_head=None,
        mlp_dim=None,
        num_patches=1,
    ):
        super().__init__()
        dim_head = dim_head or dim // heads
        mlp_dim = mlp_dim or 4 * dim
        z_dim = z_dim or dim

        self.pos_embedding = (
            nn.Parameter(torch.randn(1, 1, num_patches, dim)) if num_patches > 1 else None
        )  # position in the image, when a state is a grid of tokens
        self.transformer = Transformer(dim, dim, dim, depth, heads, dim_head, mlp_dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.readout = CrossBlock(dim, heads, dim_head, mlp_dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * z_dim)  # z_dim < dim: a bottleneck on top of the KL

        nn.init.zeros_(self.head.weight)  # start at the prior: z ~ N(0, 1), kl = 0
        nn.init.zeros_(self.head.bias)

    def forward(self, emb):
        """
        emb: (B, T, D) states of one token, or (B, T, N, D) of N tokens
        returns z: (B, T-1, z_dim), kl: scalar
        """
        mu, logvar = self.posterior(emb)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return z, kl

    def posterior(self, emb):
        """mu, logvar of the latent action of each transition: (B, T-1, z_dim) each"""
        b = emb.size(0)
        x = emb[:, :, None] if emb.dim() == 3 else emb + self.pos_embedding  # (B, T, N, D)
        n = x.size(2)
        x = torch.cat([x[:, :-1], x[:, 1:]], dim=2)  # pairs (t, t+1): (B, T-1, 2N, D)
        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.transformer(x, causal=False, frame_size=n)  # joint attention, frames 0 / 1
        query = self.query.expand(x.size(0), -1, -1)
        h = self.readout(query, x, frame_size=n)[:, 0]  # query at frame index 2

        mu, logvar = self.head(self.norm(h)).chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 10)
        return tuple(rearrange(v, "(b t) d -> b t d", b=b) for v in (mu, logvar))


class ActionEncoder(nn.Module):
    """The real actions as the predictor's conditioning, in place of the IDM: an action-conditioned world model, as
    LeWorldModel (its Embedder: a linear map then an MLP, here the MLP on the standardized action). No KL: kl = 0"""

    def __init__(self, *, act_dim, z_dim, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or 4 * z_dim
        # standardization of the real actions, set from the training actions and saved with the weights
        self.register_buffer("act_mean", torch.zeros(act_dim))
        self.register_buffer("act_std", torch.ones(act_dim))
        self.net = nn.Sequential(nn.Linear(act_dim, mlp_dim), nn.SiLU(), nn.Linear(mlp_dim, z_dim))

    @torch.no_grad()
    def set_action_stats(self, actions):
        """actions: (..., act_dim) training actions, in the dataset's units"""
        a = actions.reshape(-1, actions.size(-1)).float()
        std = a.std(0)
        self.act_mean.copy_(a.mean(0))
        self.act_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))  # constant dims stay unscaled

    def forward(self, actions):
        """actions: (B, T-1, act_dim) real actions of the transitions -> (B, T-1, z_dim)"""
        return self.net((actions.float() - self.act_mean) / self.act_std)
