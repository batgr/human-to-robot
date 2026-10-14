"""Frozen pretrained image encoder"""

import torch
from torch import nn
from transformers import AutoModel


class FrozenEncoder(nn.Module):
    """Frames in [0, 255] -> tokens of a frozen pretrained ViT, CLS first"""

    def __init__(self, name, image_size, dim):
        super().__init__()
        self.model = AutoModel.from_pretrained(name).eval().requires_grad_(False)
        config = self.model.config
        assert (
            config.hidden_size == dim
        ), f"{name} gives {config.hidden_size}-d tokens, not {dim}"
        assert (
            image_size % config.patch_size == 0
        ), f"{image_size} px is not a multiple of {config.patch_size}"
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )  # ImageNet
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        return super().train(False)  # always in eval mode

    @torch.no_grad()
    def forward(self, pixels, **kwargs):
        """
        pixels: (B, 3, H, W) in [0, 255]
        returns the model output: .last_hidden_state (B, 1 + N, dim)
        """
        x = (pixels / 255 - self.mean) / self.std
        return self.model(pixel_values=x)
