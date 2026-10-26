"""Frozen pretrained image encoder"""

import torch
from torch import nn
from transformers import AutoModel, ViTConfig, ViTModel


class FrozenEncoder(nn.Module):

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

        x = (pixels / 255 - self.mean) / self.std
        return self.model(pixel_values=x)


class ViTEncoder(nn.Module):
    """ViT trained from scratch with the world model, as in LeWorldModel (ViT-tiny, patch 14)"""

    def __init__(self, image_size, patch_size, dim, depth, heads):
        super().__init__()
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=4 * dim,
        )
        self.model = ViTModel(config, add_pooling_layer=False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )  # ImageNet
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pixels, **kwargs):
        x = (pixels / 255 - self.mean) / self.std
        return self.model(pixel_values=x, **kwargs)
