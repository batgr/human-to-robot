# Adapted from LeWorldModel (https://github.com/lucas-maes/le-wm)
# Copyright (c) 2026 Lucas Maes — MIT License, see LICENSES/LeWM-MIT.txt
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


def rope(x, pos, base=1000):
    """
    Rotary position embedding
    x : (B, heads, L, dim_head)
    pos : (L,) position of each token
    """
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, device=x.device) / d)

    ang = pos[:, None] * theta  # (L, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, frame_size=1):
        """
        x : (B, L, D), L = T * frame_size tokens (frame_size tokens per frame)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, L, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        frame = torch.arange(x.size(1), device=x.device) // frame_size  # frame index per token
        q, k = rope(q, frame), rope(k, frame)
        mask = frame[:, None] >= frame[None, :] if causal else None  # see all tokens of past frames
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Scaled dot-product attention of queries over a context"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context, frame_size=1):
        """
        x : (B, Lq, D) queries, placed right after the last context frame
        context : (B, L, D), frame_size tokens per frame
        """
        drop = self.dropout if self.training else 0.0
        q = self.to_q(self.norm_q(x))
        k, v = self.to_kv(self.norm_kv(context)).chunk(2, dim=-1)
        q, k, v = (
            rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in (q, k, v)
        )
        frame = torch.arange(context.size(1), device=x.device) // frame_size  # frame index per context token
        q_frame = frame.new_full((x.size(1),), context.size(1) // frame_size)  # queries come after the context
        q, k = rope(q, q_frame), rope(k, frame)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, frame_size=1):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), frame_size=frame_size
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, causal=True, frame_size=1):
        x = x + self.attn(self.norm1(x), causal=causal, frame_size=frame_size)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossBlock(nn.Module):
    """Transformer block where queries read a context through cross-attention"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, context, frame_size=1):
        x = x + self.attn(x, context, frame_size=frame_size)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        cond_dim=None,
    ):
        super().__init__()
        cond_dim = cond_dim or input_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim)
            if cond_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )
        self.conditional = (
            block_class is ConditionalBlock
        )  # AdaLN blocks read c, the others a causal flag

    def forward(self, x, c=None, causal=True, frame_size=1):
        x = self.input_proj(x)
        if self.conditional:
            c = self.cond_proj(c)
        for block in self.layers:
            x = (
                block(x, c, frame_size)
                if self.conditional
                else block(x, causal, frame_size)
            )
        return self.output_proj(self.norm(x))


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.BatchNorm1d,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class PatchProjector(nn.Module):
    """Compress the encoder patch grid: 16x fewer patches (two stride-2 convs, 16 x 16 -> 4 x 4 at 224 px, as in
    AD-E2E-JEPA), then average pooled to num_patches tokens if fewer (4: a 2 x 2 grid). BatchNorm on the hidden
    layer as in the LeWM projector"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_patches=16):
        super().__init__()
        self.grid = int(num_patches**0.5)
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        x: (B*T, N, D) patch tokens of a square grid, without CLS
        """
        h = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=h)
        x = F.adaptive_avg_pool2d(self.net(x), self.grid)
        return rearrange(x, "b d h w -> b (h w) d")


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        cond_dim=None,
        num_patches=1,
    ):
        super().__init__()
        self.pos_embedding = (
            nn.Parameter(torch.randn(1, 1, num_patches, input_dim))
            if num_patches > 1
            else None
        )  # position in the image, when a state is a grid of tokens
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            cond_dim=cond_dim,
        )

    def forward(self, x, c):
        """
        x: (B, T, d) states of one token, or (B, T, N, d) of N tokens (time positions by RoPE)
        c: (B, T, cond_dim) latent actions
        """
        if x.dim() == 3:
            return self.transformer(self.dropout(x), c)
        n = x.size(2)
        x = rearrange(x + self.pos_embedding, "b t n d -> b (t n) d")
        c = repeat(c, "b t d -> b (t n) d", n=n)  # same action for every token of a frame
        x = self.transformer(self.dropout(x), c, frame_size=n)
        return rearrange(x, "b (t n) d -> b t n d", n=n)
