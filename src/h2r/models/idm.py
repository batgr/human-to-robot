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
        depth=2,
        heads=4,
        dim_head=None,
        mlp_dim=None,
    ):
        super().__init__()
        dim_head = dim_head or dim // heads
        mlp_dim = mlp_dim or 4 * dim

        self.transformer = Transformer(dim, dim, dim, depth, heads, dim_head, mlp_dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.readout = CrossBlock(dim, heads, dim_head, mlp_dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * dim)  # latent actions live in the latent dim

        nn.init.zeros_(self.head.weight)  # start at the prior: z ~ N(0, 1), kl = 0
        nn.init.zeros_(self.head.bias)

    def forward(self, emb):
        """
        emb: (B, T, D)
        returns z: (B, T-1, D), kl: scalar
        """
        b = emb.size(0)
        x = torch.stack([emb[:, :-1], emb[:, 1:]], dim=2)  # pairs (t, t+1): (B, T-1, 2, D)
        x = rearrange(x, "b t two d -> (b t) two d")
        x = self.transformer(x, causal=False)  # joint attention, positions 0 / 1
        query = self.query.expand(x.size(0), -1, -1)
        h = self.readout(query, x)[:, 0]  # query at position 2

        mu, logvar = self.head(self.norm(h)).chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 10)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return rearrange(z, "(b t) d -> b t d", b=b), kl
