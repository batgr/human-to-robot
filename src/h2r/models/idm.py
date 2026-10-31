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
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return rearrange(z, "(b t) d -> b t d", b=b), kl
