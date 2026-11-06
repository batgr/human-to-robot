# Adapted from LeWorldModel (https://github.com/lucas-maes/le-wm)
# Copyright (c) 2026 Lucas Maes — MIT License, see LICENSES/LeWM-MIT.txt
"""JEPA Implementation"""

from pathlib import Path

import torch
import yaml
from einops import rearrange
from torch import nn

from h2r.models.encoder import FrozenEncoder, ViTEncoder
from h2r.models.idm import IDM
from h2r.models.module import MLP, ARPredictor, PatchProjector


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        idm,
        projector=None,
        pred_proj=None,
        history_size=3,
        num_patches=1,
        residual=False,
    ):
        super().__init__()

        self.residual = residual  # predict s_t+1 - s_t, added to s_t
        self.num_patches = num_patches  # tokens per state: 1 = the CLS token, else the encoder's patch tokens

        self.history_size = history_size  # states seen by the predictor when unrolling
        self.encoder = encoder
        self.predictor = predictor
        self.idm = idm  # latent actions, fed directly to the predictor
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, pixels):
        """Encode observations into embeddings.
        pixels: (B, T, 3, H, W) in [0, 255]
        returns emb: (B, T, D) from the CLS token, or (B, T, N, D) from the patch tokens
        """

        pixels = pixels.float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        tokens = self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state
        if self.num_patches == 1:
            emb = self.projector(tokens[:, 0])  # cls token: (B*T, D)
        else:  # patch tokens (drop cls): every one through the MLP, or compressed by the patch projector
            emb = self.projector(tokens[:, 1:])  # (B*T, N, D)
        return rearrange(emb, "(b t) ... -> b t ...", b=b)

    def predict(self, emb, z):
        """Predict next state embedding
        emb: (B, T, D) or (B, T, N, D)
        z: (B, T, z_dim) latent actions
        """
        preds = self.predictor(emb, z)
        preds = self.pred_proj(preds.flatten(0, -2)).view_as(preds)
        return emb + preds if self.residual else preds

    ####################
    ## Inference only ##
    ####################

    @torch.no_grad()
    def rollout(self, emb, latents, actions, controller=None):
        """Imagine the future of real action sequences, the controller in the loop (Garrido et al. 2026, eq. 2)
        emb: (B, C, D) or (B, C, N, D) context states, the last one is the current state
        latents: (B, C - 1, z_dim) latent actions linking the context states
        actions: (B, S, H, act_dim) S sequences of H real actions, or of H latent actions when there is
            no controller (e.g. the IDM's, to evaluate the predictor alone)
        returns the predicted states (B, S, H, D) or (B, S, H, N, D)
        """
        b, s, h = actions.shape[:3]
        emb = emb[:, None].expand(-1, s, *emb.shape[1:]).flatten(0, 1)  # (B*S, C, D)
        z = latents[:, None].expand(-1, s, -1, -1).flatten(0, 1)  # (B*S, C - 1, z_dim)
        actions = actions.flatten(0, 1)
        for i in range(h):
            # the controller reads the state the action starts from (real, then predicted); z[k] leads from emb[k]
            step = (
                controller(actions[:, i], emb[:, -1].flatten(1))
                if controller
                else actions[:, i]
            )
            z = torch.cat([z, step[:, None]], dim=1)
            k = self.history_size
            pred = self.predict(emb[:, -k:], z[:, -k:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return emb[:, -h:].unflatten(0, (b, s))


ENCODERS = {"frozen": FrozenEncoder, "vit": ViTEncoder}


def build_model(cfg):
    enc = dict(cfg["encoder"])
    encoder = ENCODERS[enc.pop("type")](**enc)
    n = cfg.get("num_patches", 1)
    grid = (enc["image_size"] // encoder.model.config.patch_size) ** 2  # encoder patch tokens: 256 at 224 px
    projector = (
        MLP(input_dim=enc["dim"], **cfg["projector"])  # on the CLS token (n = 1) or on every patch token (n = grid)
        if n in (1, grid)
        else PatchProjector(input_dim=enc["dim"], num_patches=n, **cfg["projector"])
    )
    pred_proj = MLP(**cfg["pred_proj"])
    residual = cfg.get("residual", False)
    if residual:  # the change is 0 at init: the model starts as persistence (test.ipynb, 30 Oct.)
        nn.init.zeros_(pred_proj.net[-1].weight)
        nn.init.zeros_(pred_proj.net[-1].bias)
    return JEPA(
        encoder,
        ARPredictor(num_patches=n, **cfg["predictor"]),
        IDM(num_patches=n, **cfg["idm"]),
        projector,
        pred_proj,
        history_size=cfg.get("history_size", 3),  # runs saved before the key: 3
        num_patches=n,
        residual=residual,
    )


def load_world_model(run_dir, device, checkpoint="best.pt"):
    """Frozen world model of a training run (config.yaml + a checkpoint)"""
    run_dir = Path(run_dir)
    if not (run_dir / checkpoint).exists():
        raise FileNotFoundError(f"world_model: no {checkpoint} in {run_dir}")
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(
        torch.load(run_dir / checkpoint, map_location=device, weights_only=False)[
            "model"
        ],
        strict=False,
    )
    model.eval().requires_grad_(False)
    return model, cfg
