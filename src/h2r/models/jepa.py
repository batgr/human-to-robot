# Adapted from LeWorldModel (https://github.com/lucas-maes/le-wm)
# Copyright (c) 2026 Lucas Maes — MIT License, see LICENSES/LeWM-MIT.txt
"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


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

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, D) latent actions
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            emb_trunc = emb[:, -HS:]  # (BS, HS, N, D)
            act_trunc = act[:, -HS:]  # (BS, HS, D)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, N, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, N, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        emb_trunc = emb[:, -HS:]  # (BS, HS, N, D)
        act_trunc = act[:, -HS:]  # (BS, HS, D)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, N, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"][:, :, -1]  # last step: (B, S, N, D)
        goal_emb = info_dict["goal_emb"][:, -1]  # last goal frame: (B, N, D)
        goal_emb = goal_emb.unsqueeze(1).expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(pred_emb, goal_emb.detach(), reduction="none").sum(
            dim=(2, 3)
        )  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
