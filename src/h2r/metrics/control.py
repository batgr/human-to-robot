"""Control: the world model driven by real actions through the controller, the measures of Garrido et al. 2026
(section 5, appendix A, tables S1-S2)"""

import torch


class ControlledRollout:
    """Open-loop unrolling from one real state (JEPA-WM's protocol, as RolloutError), with the latent actions of the
    controller from the true real actions (ctrl) and of the IDM (idm, a reference: it sees the next frame).
    - error_h{h}, error_idm_h{h}: MSE of the state h steps ahead;
    - error_ratio_h{h}: error with the controller / error with the IDM, the measure of Garrido et al. 2026 (tables S1-S2:
      x1.0-1.5 on DROID and RECON, in LPIPS of decoded frames: not a threshold for this embedding MSE)."""

    def __init__(self):
        self.err = {"ctrl": [], "idm": []}

    def update(self, model, ctrl, emb, mu, actions):
        """emb: (B, H + 1, D) or (B, H + 1, N, D) states of an episode window, mu: (B, H, z_dim) the IDM's latent actions
        of its transitions, actions: (B, H, act_dim) the real ones"""
        start, past, target = emb[:, :1], mu[:, :0], emb[:, 1:].float()
        preds = {
            "ctrl": model.rollout(start, past, actions[:, None], ctrl)[:, 0],
            "idm": model.rollout(start, past, mu[:, None])[:, 0],
        }
        for k, pred in preds.items():
            self.err[k].append((pred.float() - target).pow(2).flatten(2).mean(-1).cpu())  # (B, H)

    def compute(self):
        err = {k: torch.cat(v).double().mean(0) for k, v in self.err.items()}
        out = {}
        for h in range(len(err["ctrl"])):
            out |= {
                f"error_h{h + 1}": float(err["ctrl"][h]),
                f"error_idm_h{h + 1}": float(err["idm"][h]),
                f"error_ratio_h{h + 1}": float(err["ctrl"][h] / err["idm"][h]),
            }
        return out
