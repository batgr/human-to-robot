# Adapted from LeWorldModel (https://github.com/lucas-maes/le-wm)
# Copyright (c) 2026 Lucas Maes — MIT License, see LICENSES/LeWM-MIT.txt
"""JEPA Implementation"""

import torch
from einops import rearrange
from torch import nn


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        idm,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.idm = idm  # latent actions, fed directly to the predictor
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations into embeddings.
        info: dict with pixels key
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 1:]  # patch tokens (drop cls)
        emb = self.projector(pixels_emb)  # (B*T, N, D)
        info["emb"] = rearrange(emb, "(b t) n d -> b t n d", b=b)
        return info

    def predict(self, emb, z):
        """Predict next state embedding
        emb: (B, T, N, D)
        z: (B, T, D) latent actions
        """
        b, t = emb.shape[:2]
        preds = self.predictor(emb, z)
        preds = self.pred_proj(rearrange(preds, "b t n d -> (b t n) d"))
        preds = rearrange(preds, "(b t n) d -> b t n d", b=b, t=t)
        return preds

    ####################
    ## Inference only ##
    ####################

    @torch.no_grad()
    def rollout(self, emb, latents, actions, controller, history_size=3):
        """Imagine the future of real action sequences, the controller in the loop (Garrido et al. 2026, eq. 2)
        emb: (B, C, N, D) context states, the last one is the current state
        latents: (B, C - 1, D) latent actions linking the context states
        actions: (B, S, H, act_dim) S candidate sequences of H real actions
        returns the predicted states (B, S, H, N, D)
        """
        b, s, h = actions.shape[:3]
        emb = emb[:, None].expand(-1, s, -1, -1, -1).flatten(0, 1)  # (B*S, C, N, D)
        z = latents[:, None].expand(-1, s, -1, -1).flatten(0, 1)  # (B*S, C - 1, D)
        actions = actions.flatten(0, 1)
        for i in range(h):
            # the controller reads the state the action starts from (real, then predicted); z[k] leads from emb[k]
            z = torch.cat([z, controller(actions[:, i], emb[:, -1])[:, None]], dim=1)
            pred = self.predict(emb[:, -history_size:], z[:, -history_size:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return emb[:, -h:].unflatten(0, (b, s))

    def criterion(self, pred, goal):
        """Squared L2 distance of the last predicted state to the goal: (B, S, H, N, D), (B, N, D) -> (B, S)"""
        return (pred[:, :, -1] - goal[:, None]).pow(2).sum(dim=(-2, -1))

    def get_cost(self, emb, latents, goal, actions, controller, history_size=3):
        """Cost of each candidate action sequence: (B, S)"""
        return self.criterion(self.rollout(emb, latents, actions, controller, history_size), goal)
