# Human-to-Robot

**Learning latent actions and visual dynamics from unlabeled manipulation videos.**

Can an action-conditioned world model trained on human videos transfer to a robot with a different embodiment? We train an inverse dynamics model (IDM) and a JEPA-style predictor on videos without action labels, then test the learned dynamics on a Panda arm in Robosuite.

The current models learn some in-domain dynamics but do **not** yet transfer successfully to simulation. This repository includes the data, checkpoints, experiments, and failure cases.

**Resources:** [iPhone videos](https://huggingface.co/datasets/batgre/h2r-phone) · [Robot videos](https://huggingface.co/datasets/batgre/h2r-robots-1h30) · [Simulation episodes](https://huggingface.co/datasets/batgre/h2r-sim-v3) · [Checkpoints](https://huggingface.co/batgre/h2r-lift)

## Run the simulation

Run the pretrained **human-video** model and controller on five held-out Robosuite Panda Lift episodes. No training is needed.

```bash
uv sync

uv run hf download batgre/h2r-lift --local-dir outputs \
  --include "wm-phone-h2/*" --include "ctrl-phone-h2/*"

uv run hf download batgre/h2r-sim-v3 --repo-type dataset \
  --local-dir data/h2r-sim --include "raw/Panda/Lift/scripted/*"

uv run h2r eval pick --set name=demo-phone-h2 \
  controller=outputs/ctrl-phone-h2 "policies=[planner]" \
  episodes=5 planner.horizon=2
```

Results: `outputs/demo-phone-h2/results.json` and an episode filmstrip (`episode_failure.png` for the first failure; `episode_success.png` only if a pick succeeds). This is a **headless evaluation**, not an interactive GUI or an MP4 recording. The published planners achieved **0/30 successful lifts**.

## Model

The IDM infers what changed between consecutive encoded frames. The predictor uses those latent actions to predict the next visual state.

```text
video frame ──> frozen DINOv2-small ──> projector ──> state s_t
                                                   │
                         (s_t, s_t+1) ──> IDM ──> latent action z_t
                                                   │
                       past states + actions ──> predictor ──> next state
```

| Part | Implementation |
|---|---|
| Visual state | Frozen DINOv2-small; learned projector to 16 × 128-D patch tokens |
| IDM | 2-layer Transformer over two consecutive frames; Gaussian latent action (128-D) |
| Predictor | 6-layer causal Transformer; AdaLN action conditioning |
| Rollout | Predict in latent space, then feed predicted states into subsequent steps |

**2.94M trainable parameters**, excluding DINOv2. The IDM and predictor are trained together, using videos only. Neither needs robot actions, pose labels, or object annotations.

Code: [IDM](src/h2r/models/idm.py) · [World model](src/h2r/models/jepa.py) · [Predictor](src/h2r/models/module.py)

## Data

| Source | Contents | Used for |
|---|---|---|
| [iPhone videos](https://huggingface.co/datasets/batgre/h2r-phone) | 158 personally recorded manipulation videos, 20.5 min, two camera angles (124 / 14 / 20 train / val / test) | Human-video world model |
| [Real-robot videos](https://huggingface.co/datasets/batgre/h2r-robots-1h30) | 173 videos selected from six Open-X/LeRobot corpora for Run 2 | Robot-video world model |
| [Robosuite episodes](https://huggingface.co/datasets/batgre/h2r-sim-v3) | Panda Lift; 1,100 scripted and 1,100 random episodes with states and actions | Controller training and evaluation |

Frames are sampled at **4 fps**, resized to **224 × 224**, and split by video. Run 2 matches the number of training frames (4,181 human vs 4,183 robot); clip counts and motion statistics are not identical. The iPhone footage was collected from one person. Audio and location metadata are excluded from the public dataset.

The two world models see **only their respective real-world videos** during pretraining; simulated actions are used later for the controller.

## Training

We jointly optimize the IDM and predictor with:

```text
loss = teacher-forced MSE + rollout MSE + 0.09 × SIGReg + 0.001 × KL
```

SIGReg regularizes the visual states; KL regularizes the IDM's Gaussian action distribution. Rollout training starts at one step and increases its horizon during the first half of the run. Predicted-state inputs are detached between rollout steps.

Both experiments use a 3-frame context, batch size 128, 1,000 optimizer steps, Muon + AdamW, and cosine scheduling. Checkpoints are selected by mean validation prediction skill.

| | Run 1 | Run 2 |
|---|---:|---:|
| Prediction horizon | 5 steps | 2 steps |
| iPhone video | 12.2 min, one angle | 20.5 min, two angles |
| Real-robot videos | 102 | 173 |

Run 1 and Run 2 change several variables at once, so their difference is **not a controlled ablation**.

Code: [training loop](src/h2r/training/world_model.py) · [model config](configs/models/world_model.yaml)

## Results

### 1. Prediction on held-out videos

We report `skill(h) = 1 - prediction_MSE / persistence_MSE`, where persistence repeats the last observed latent state. **Positive skill** means the prediction beats persistence.

The future actions used in this diagnostic are **inferred by the IDM from real frame pairs**, including future observations. This measures prediction conditioned on inferred actions, not autonomous action selection.

| World model | Horizon | Skill: step 1 / final | Action sensitivity | Active IDM dimensions |
|---|---:|---:|---:|---:|
| Run 1 · iPhone | 5 | -0.07 / 0.15 | 0.26 | 127 / 128 |
| Run 1 · robot | 5 | -0.66 / -0.64 | 0.06 | 74 / 128 |
| Run 2 · iPhone | 2 | **0.17 / 0.29** | **0.40** | 89 / 128 |
| Run 2 · robot | 2 | -0.49 / -0.52 | 0.09 | 71 / 128 |

Action sensitivity compares error with correct versus shuffled IDM actions. The robot-video predictor depends much less on its actions, and its IDM shows partial action collapse. A high sensitivity score **does not** mean latent actions are physically meaningful.

Persistence is harder to beat on videos with little movement. Values measured on different datasets and model-specific latent spaces are therefore not a direct comparison of absolute prediction quality.

Training curves: [iPhone model](docs/figures/curves-wm-phone-h2.png) · [robot model](docs/figures/curves-wm-robots-h2.png)

### 2. Transfer to other video domains

Run 2 models are evaluated without fine-tuning. Each entry is skill at step 1 / step 2, using IDM-inferred future actions.

| Trained on | iPhone validation | Robot validation | Robosuite |
|---|---:|---:|---:|
| iPhone | 0.17 / 0.29 | -1.89 / -2.49 | -2.14 / -2.45 |
| Robot | -0.78 / -1.03 | -0.49 / -0.52 | -2.31 / -2.49 |

**Neither model beats persistence in Robosuite.** The learned dynamics remain strongly tied to the training domain.

The images below compare real future frames (top) with the real dataset frames nearest to the *predicted latent states* (bottom). They are **nearest-neighbor visualizations**, not decoded model outputs.

| Trained on iPhone videos | Trained on robot videos |
|:---:|:---:|
| ![Real vs nearest predicted frames: iPhone-trained WM](docs/figures/imagined-phone-h2.png) | ![Real vs nearest predicted frames: robot-trained WM](docs/figures/imagined-robots-h2.png) |

### 3. Downstream simulation

A separate controller maps robot end-effector actions into the IDM's latent-action space. CEM plans sequences in Robosuite, executes the first action, then observes and replans. The planner is given pre-grasp, grasp and lift goal images taken from held-out scripted demonstrations.

**Neither planner lifted the cube in 30 test episodes.** Random actions also scored 0%; replaying recorded demonstration actions succeeded in 73–87% across three evaluations. Capping each planned step at 2 cm (`planner.max_norm=0.02`) did not change this: 0 of 30 for both models. The learned model is not yet accurate enough under the simulation's action and visual distributions.

![Example failed pick with the iPhone-video world model](docs/figures/pick-phone-h2-failure.png)

[See the robot-video model's failure](docs/figures/pick-robots-h2-failure.png).

## Analysis

**Where the pipeline breaks.** We ran diagnostics on held-out Robosuite episodes for the Run 2 models. Each diagnostic checks one link of the chain, from prediction to action ([`src/h2r/eval/`](src/h2r/eval)):

| Run 2 | iPhone WM | Robot WM |
|---|---:|---:|
| Prediction skill in Robosuite, step 1 | -3.48 | -3.98 |
| Prediction follows the command: with IDM actions / through the controller | 0.22 / -0.01 | 0.22 / 0.15 |
| Offline planning error Δxyz: planner / no motion | 10.9 / 2.8 cm | 33.3 / 2.8 cm |
| Same, steps capped at 2 cm | 4.6 / 2.8 cm | 5.3 / 2.8 cm |

*"Follows the command": cosine between predicted and real differences of outcomes for several command sequences played from the same state; 0 means unrelated. Offline planning follows Garrido et al.: 64 clips, the real frame 2 steps ahead as the goal, and the L1 error between planned and recorded displacement.*

1. **Prediction.** Neither model has seen a simulated frame, and both predict Robosuite frames worse than repeating the last one.
2. **Control.** The iPhone model's predictions respond to the command when the IDM gives the latent actions. Through the controller they no longer do, so the robot action's effect is lost in the controller. The robot model's controller keeps part of it.
3. **Planning.** Every planner does worse than not moving: CEM picks displacements much larger than in the demonstrations. A 2 cm cap per step reduces the error but does not fix it.
4. **Task length.** The lift is about 30 steps away; the planner looks 2 steps ahead, guided by subgoals.

The action interface is not the cause: replaying the recorded actions through the same pipeline lifts the cube in 73–87% of episodes.

**What the two runs suggest.**
- **Data is the main limit.** Going from 12.2 to 20.5 min of iPhone video and adding a second camera angle came with better prediction on every split. The horizon also changed (5 to 2 steps), so this is not a controlled ablation. The one-step validation error is 1.7× the training error for the iPhone model and 2.0× for the robot model, so both models overfit their few videos. For scale, Garrido et al. train on YouTube-Temporal-1B with about 240× more clips and a ViT-L predictor.
- **More robot video is not automatically better.** The 173 robot videos come from 6 datasets with different robots, cameras and scenes. That model never beats persistence on its own videos, overfits after step 650 and its latent action partly collapsed.
- **Good offline diagnostics did not predict a pick.** The Run 1 robot model had the best controller and ranking scores and still picked no cube.

**Next steps**, in order of expected impact:
1. **Much more human video.** Something-Something v2 (220k clips of hands manipulating objects), then Ego4D.
2. **A little in-domain video.** Garrido et al. (Table S3) show that 10% in-domain data cuts their planning error from 0.14 to 0.09. Our simulated episodes already have videos.
3. **Fix the controller–planner link.** Retrain the iPhone model's controller, bound the action search, and test reaching before the full pick.
4. **An action-labelled baseline.** A world model trained on the simulated episodes with their real actions would show how much of the gap comes from learning actions from video.

## Checkpoints

[All checkpoints and configs on Hugging Face](https://huggingface.co/batgre/h2r-lift).

| Model | Human / iPhone | Real robot |
|---|---|---|
| World model · H=5 | [wm-phone](https://huggingface.co/batgre/h2r-lift/tree/main/wm-phone) | [wm-robots](https://huggingface.co/batgre/h2r-lift/tree/main/wm-robots) |
| World model · H=2 | [wm-phone-h2](https://huggingface.co/batgre/h2r-lift/tree/main/wm-phone-h2) | [wm-robots-h2](https://huggingface.co/batgre/h2r-lift/tree/main/wm-robots-h2) |
| Controller · H=2 | [ctrl-phone-h2](https://huggingface.co/batgre/h2r-lift/tree/main/ctrl-phone-h2) | [ctrl-robots-h2](https://huggingface.co/batgre/h2r-lift/tree/main/ctrl-robots-h2) |

## Reproduce training

The [Colab notebook](lift.ipynb) runs data preparation, training and evaluation. To repeat the Run 2 world-model experiments:

```bash
uv sync

uv run hf download batgre/h2r-phone --repo-type dataset \
  --revision 815bfb769aa6fc2059fe746381473915a2fb665d \
  --local-dir data/h2r-phone --exclude "raw/*"

uv run hf download batgre/h2r-robots-1h30 --repo-type dataset \
  --local-dir data/h2r-robots-1h30

H="rollout_steps.end=2 val_horizon=2 iterations=1000 eval_every=50"
uv run h2r train --set name=wm-phone-h2 \
  data.root=data/h2r-phone data.hf_repo=batgre/h2r-phone $H

uv run h2r train --set name=wm-robots-h2 \
  data.root=data/h2r-robots-1h30 \
  data.hf_repo=batgre/h2r-robots-1h30 data.max_videos=173 $H
```

Use `uv run h2r controller train` to fit the downstream action controller. Evaluation and data commands are listed in `uv run h2r --help`, and their configs are in [`configs/`](configs/). Run outputs include checkpoints, metrics and the resolved configuration. Tests: `uv run pytest`.

## Limitations

- Only **20.5 minutes of personally recorded videos** from one person, with two camera angles.
- One training seed per model; downstream results use 30 held-out simulation episodes.
- Evaluation prediction actions are inferred from future frames and are not directly usable by a robot.
- Run 1 vs Run 2 changes multiple factors together.
- Domain shift and weak action conditioning prevent successful robot control in the current experiments.

## References and credits

Parts of [`jepa.py`](src/h2r/models/jepa.py) and [`module.py`](src/h2r/models/module.py) are adapted from [LeWorldModel](https://github.com/lucas-maes/le-wm), MIT license ([license copy](LICENSES/LeWM-MIT.txt)). The IDM and related training choices follow Garrido et al.

- Maes et al., [*LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*](https://arxiv.org/abs/2603.19312), 2026.
- Garrido et al., [*Learning Latent Action World Models in the Wild*](https://arxiv.org/abs/2601.05230), 2026.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2024.
- Terver et al., *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?*, 2026.
