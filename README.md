# Human-to-Robot

**Learning latent actions and visual dynamics from unlabeled human videos, then planning with them on a robot.**

Can an action-conditioned world model trained on human videos, without action labels, drive a robot with a different embodiment? We train a ViT encoder from scratch, an inverse dynamics model (IDM) and a JEPA-style predictor on 20 h of Something-Something v2. Robot actions enter through a small controller that maps them to the model's latent actions, and a CEM planner uses the model on a Panda arm in Robosuite.

So far: the model predicts human video well, and it follows robot paths when its latent actions are inferred from the frames. The robot's own actions drive it much less.

![Closed-loop planning in robosuite: robot planned by the model (top) and goal image (bottom)](docs/figures/closed-loop-planning.png)

*Closed-loop planning with the 20 h model: at each step the CEM plans the next action toward the next frame of a demonstration, the action is executed, and the model replans from the new image. This is the best episode (validation episode 20): reach at step 13, grasp at 25, lift at 26; the cube drops at step 30. It is the only lift out of 20 episodes in this setting.*

## Model

![Architecture: training on human video (left) and planning on the robot (right)](docs/figures/architecture.svg)

The reference world model is the 20 h run `wm-cls-ssv2-20hall-k2from1050-b3e-4-20k`, checkpoint `step-20000.pt`, trained at commit `248588a`.

| Part | Specification | Parameters |
|---|---|---:|
| Encoder | ViT-tiny from scratch: 224 px, patch 14, 192-d, 12 layers, 3 heads | 5.50 M |
| Projector | MLP on the CLS token: 192 → 2048 → 192, BatchNorm, GELU | 0.79 M |
| IDM | Pairs (s_t, s_t+1), 2 transformer layers, 4 heads, read by one query; Gaussian z, 64-d | 1.36 M |
| Predictor | 6 layers, causal, 16 heads of 64, MLP 2048, AdaLN-zero on z, RoPE on time; predicts the next state directly | 10.80 M |
| Prediction projector | MLP: 192 → 2048 → 192, BatchNorm, GELU | 0.79 M |
| **Total** | | **19.25 M** |

| Training | Value |
|---|---|
| Data | SSv2 20 h: 19,301 videos, 20.9 h, 168 classes, 4 fps, no action labels |
| Clips | 8 consecutive frames (2 s), 7 transitions |
| Loss | teacher forcing + 2-step rollout from step 1,050 (last-gradient only, 3 states of context) + β KL + λ SIGReg |
| Weights | β 3e-4, λ 0.09 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-3, 10% warmup, gradient clip 1.0, bf16 |
| Length | 20,000 steps, batch 128, seed 0 |

## Results

All numbers are for the model above, each beside a reference. **Persistence** is the forecast that nothing changes (ŝ_t+1 = s_t); prediction errors are given as a fraction of its error, so below 1 beats it.

### Human video (held-out SSv2)

| Measure | Model | Reference |
|---|---|---|
| Next-state error / persistence | 0.0135 | 1 |
| 7-step rollout error / persistence | 0.013 | 1 |
| Motion invented on still clips (predicted / true change) | 1.09 | 1 = nothing invented |
| Controllability: how much the prediction changes when z comes from another clip | 0.99 | 0 = z ignored |
| Leak: share of predictions that copy the next frame of the clip z came from | 0.015 | 0 = no copy |
| Rank of the states (RankMe) | 72 | 192 dimensions |
| Active dimensions of z | 8 | 64 dimensions |
| Action class read from z (accuracy) | 0.015 | chance 0.006 |
| Left / right read from z (accuracy) | 0.50 | chance 0.50 |
| Hand position read from the encoder (linear probe R²) | 0.08 | frozen DINOv2: 0.39–0.43 |

**More video keeps helping.** Prediction error falls about 20 % per doubling of SSv2 video, with no saturation up to 8 h; more classes at equal hours help more (3,000-step runs).

![Validation error versus hours of video](docs/figures/experiments/data-scaling.png)

### Robot (Robosuite Lift, Panda arm)

| Measure | Model | Reference |
|---|---|---|
| Robot paths over 7 steps, cosine to the real path, z from the IDM / from the robot's actions (test, most moving third) | simulation 0.93 / 0.70 | DROID 0.83 / 0.02 |
| Gripper closure read from the state (linear probe R²) | 0.04 | frozen DINOv2-S: 0.84 |
| Hit rate: the true actions ranked first / in the top 5 by the model's cost, among 64 sequences, goal 3 steps ahead | 10.4 % / 31.9 % | chance 1.6 % / 7.8 % |
| Offline FDE: L1 error of the planned Δxyz, 3 actions toward a goal 7 steps ahead | 0.122 | zero plan 0.087, another episode's move 0.130 |
| Offline FDE, 1 action toward the next frame | 0.044 | zero plan 0.038; on the most moving third, 0.0045 better than the zero plan |
| Closed loop, 20 episodes: reach / grasp / lift, best of 7 planner settings for each stage taken separately (no single setting reaches 7 / 3 / 1) | 7 / 3 / 1 | recorded actions 18 / 19 / 16, zero plan 0 / 0 / 0 |

**The robot's path follows the frames, not the robot's actions.** One test clip, its real path in the model's state space, and the paths the model imagines from its first state: with z inferred from the frames by the IDM, and with z from the robot's real actions through the controller.

![One test clip: real path and paths imagined with the IDM and with the controller](docs/figures/experiments/clip-path.png)

*A typical clip: among the 375 most moving windows of the scripted test episodes, the closest to the median of both cosines (IDM 0.90, controller 0.44 on these windows; the table counts the random episodes too).*

## Quickstart

```bash
uv sync

# frame shards from a folder of videos (configs/data/config.yaml); our SSv2 copy on the Hub is private (licence)
uv run h2r data extract --set raw_dir=data/raw/ processed_dir=data/processed/

# world model, current recipe: 20,000 steps, 2-step rollout loss from epoch 1, β 3e-4
uv run h2r train wm --set name=<run> data.processed_dir=data/processed/ \
    loss.rollout_steps=2 loss.rollout_from_epoch=1 loss.beta=0.0003 iterations=20000

# controller (robot action -> z) on the simulated episodes, public: batgre/h2r-sim-v3
uv run h2r train controller --set world_model=outputs/<run> data.root=data/h2r-sim/ data.hf_repo=batgre/h2r-sim-v3

# closed-loop planning in robosuite Lift, dense subgoals t+1
uv run h2r plan run --set policy.controller=outputs/<controller run> policy.goal=dense policy.k=1 policy.horizon=1
```
