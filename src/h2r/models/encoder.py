"""Image encoders: frozen pretrained, ViT or BiT ResNet-v2 trained from scratch"""

import torch
from torch import nn
from transformers import AutoModel, BitConfig, BitModel, ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutput


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
        self.patch_size = config.patch_size
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
        self.patch_size = patch_size
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )  # ImageNet
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pixels, **kwargs):
        x = (pixels / 255 - self.mean) / self.std
        return self.model(pixel_values=x, **kwargs)


class BitEncoder(nn.Module):
    """BiT ResNet-v2 (Kolesnikov et al. 2020, transformers' BitModel) trained from scratch with the world model: a
    CNN's locality bias in place of the ViT. Pre-activation bottlenecks, GroupNorm + Weight Standardization.
    dim = output channels: 2048 for BiT R50x1 (23.5 M parameters), 1024 at half width (5.9 M, close to ViT-tiny's
    5.5 M). Tokens like a ViT: the global average pool (BiT's head, before the classifier) in place of the CLS, then
    the final feature map (7 x 7 at 224 px)"""

    def __init__(self, image_size, dim, depths=(3, 4, 6, 3)):
        super().__init__()
        config = BitConfig(
            embedding_size=dim // 32,
            hidden_sizes=[dim // 8, dim // 4, dim // 2, dim],
            depths=list(depths),
            layer_type="preactivation",
        )
        self.model = BitModel(config)
        self.patch_size = config.output_stride  # 32
        assert image_size % self.patch_size == 0, f"{image_size} px is not a multiple of {self.patch_size}"
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )  # ImageNet
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pixels, **kwargs):
        out = self.model(pixel_values=(pixels / 255 - self.mean) / self.std)
        x = out.last_hidden_state.flatten(2).transpose(1, 2)  # (B, H * W, dim), after the final GroupNorm + ReLU
        return BaseModelOutput(last_hidden_state=torch.cat([out.pooler_output.flatten(1)[:, None], x], dim=1))
