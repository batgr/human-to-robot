# Human-to-Robot

**Learning action-conditioned world models from unlabeled manipulation videos**

This project investigates whether a world model can learn useful action representations and visual dynamics directly from manipulation videos, without ground-truth action labels.

We jointly train an inverse dynamics model (IDM) and a JEPA-style world model. The IDM infers latent actions from consecutive visual observations; the world model predicts future representations conditioned on these actions. Multi-step autoregressive training makes the predictor learn from its own predicted states.

We compare models trained on personally recorded human videos and videos of real robots, focusing on prediction quality, action conditioning, representation collapse, and generalization across domains. Goal-conditioned planning in Robosuite is a downstream experiment, not part of world-model pretraining.

## Architecture

The IDM and predictor are trained jointly on unlabeled video. The visual encoder is frozen.

\`\`\`text
video frames ──> frozen DINOv2-small ──> learned projector ──> visual states sₜ
                                                           │
                                    (sₜ, sₜ₊₁) ──> IDM ──> latent action zₜ
                                                           │
                        (past visual states, latent actions) ──> predictor ──> ŝₜ₊₁
\`\`\`

| Component | Implementation | Output |
|---|---|---|
| **Visual encoder** | Frozen DINOv2-small; 224 × 224 inputs | Patch features per frame |
| **Projector** | Two stride-2 convolutions | 4 × 4 tokens, dimension 128 |
| **IDM** | Two-layer Transformer attending jointly to consecutive frames, learned query, Gaussian readout | One 128-dimensional latent action per transition |
| **Predictor** | Six-layer causal Transformer conditioned by latent actions through AdaLN | Next-state patch tokens |
| **Prediction projection** | Learned MLP | Predicted state in the visual latent space |

The IDM parameterizes a Gaussian latent action distribution. It samples using reparameterization during training and uses the posterior mean at evaluation. The model has approximately **2.94 million trainable parameters**, excluding the frozen visual encoder.

The predictor accepts a history of three observed or predicted visual states and the latent actions connecting them. During autoregressive rollout, each predicted state can become an input to the next step.

Implementation: [\`idm.py\`](src/h2r/models/idm.py), [\`jepa.py\`](src/h2r/models/jepa.py), [\`module.py\`](src/h2r/models/module.py).

## Training

Both models learn from **video only**. No joint states, object poses, robot commands, or action labels are used to train the IDM and world model.

The joint objective is:

\`\`\`text
L = L_teacher_forcing + L_rollout + 0.09 L_SIGReg + 0.001 L_KL
\`\`\`

| Term | Role |
|---|---|
| Teacher-forced MSE | Predict the next encoded state from observed context and IDM actions |
| Rollout MSE | Predict multiple future states using the model's own previous predictions |
| SIGReg | Discourage collapse of learned visual representations |
| KL divergence | Regularize the IDM's Gaussian action distribution |

Training starts with one-step prediction and increases the autoregressive rollout length over the first half of training. Predicted-state inputs are detached between rollout steps (truncated backpropagation through time); the inferred actions still receive gradients.

**Experimental settings:** three-frame context, 128 clips per batch, 1,000 optimizer steps, Muon for Transformer matrices and AdamW for the remaining trainable parameters, cosine learning-rate scheduling and a 10% warmup. The checkpoint with the highest mean validation prediction skill is retained.

Two experiments were run:

| | Run 1 | Run 2 |
|---|---|---|
| Rollout horizon | 5 steps | 2 steps |
| Human video | 12.2 min, one camera angle | 20.5 min, two camera angles |
| Robot video | 102 videos | 173 videos |
| Training | 1,000 steps | 1,000 steps |

Run 2 uses nearly the same number of sampled training frames for both world models (4,181 human and 4,183 robot), but their scene diversity, motion statistics, and number of usable clips differ. Run 1 and Run 2 also change several factors at once and should not be treated as a controlled ablation.

Implementation: [\`world_model.py\`](src/h2r/training/world_model.py), [model configuration](configs/models/world_model.yaml).

## Experiments & Results

We examine prediction quality, whether the predictor uses latent actions, and how the learned dynamics generalize across human, robot, and simulated-robot observations.

### Prediction on held-out videos

Prediction skill compares the multi-step MSE against a persistence baseline that repeats the last observed state:

\`\`\`text
skill(h) = 1 - MSE(predicted state at t+h, true state at t+h)
               / MSE(last context state, true state at t+h)
\`\`\`

**Positive skill** means better prediction than persistence. This evaluation uses latent actions inferred by the IDM from real future frame pairs (oracle actions). It measures the predictor and inferred-action representation, **not** the ability to choose future actions independently.

Results below are computed on each model's own held-out video split, at its selected checkpoint:

| Model | Horizon | Skill (step 1 / final step) | Controllability | Active IDM dimensions / 128 |
|---|---:|---:|---:|---:|
| Run 1, human | 5 | -0.07 / 0.15 | 0.26 | 127 |
| Run 1, robot | 5 | -0.66 / -0.64 | 0.06 | 74 |
| Run 2, human | 2 | **0.17 / 0.29** | **0.40** | 89 |
| Run 2, robot | 2 | -0.49 / -0.52 | 0.09 | 71 |

In Run 2, the human-video model outperforms persistence on its held-out videos, while the robot-video model does not. Their absolute one-step validation MSE values are similar (0.152 and 0.154), but these losses are measured on different datasets and model-specific representation spaces. Persistence is harder to beat on robot videos with little frame-to-frame motion. Skill values across these datasets therefore do not provide a direct model-quality ranking.

![Run 2: human-video world model training curves](docs/figures/curves-wm-phone-h2.png)

### Does the predictor use the latent action?

A predictor may reduce its loss while ignoring the action input. We measure **controllability** by comparing teacher-forced prediction error with the correct IDM actions against error after shuffling actions across clips. A value near zero indicates little sensitivity to this intervention.

At the selected Run 2 checkpoints, controllability is **0.40** on human videos and **0.09** on robot videos. The robot-video IDM also shows partial action collapse during training: its active action dimensions drop to zero early in Run 2 and recover only partially.

Action sensitivity is necessary but insufficient evidence of useful control. It does not establish that individual latent dimensions have stable physical meanings.

### Cross-domain generalization

Each Run 2 model is evaluated without fine-tuning on held-out human videos, real-robot videos, and Robosuite observations.

| World model trained on | Human test videos | Real-robot test videos | Robosuite |
|---|---|---|---|
| Human | 0.17 / 0.29 | -1.89 / -2.49 | -2.14 / -2.45 |
| Robot | -0.78 / -1.03 | -0.49 / -0.52 | -2.31 / -2.49 |

*Entries are prediction skill at steps 1 / 2, using IDM-inferred actions.*

Neither model outperforms persistence on simulated robot observations. Controllability also drops substantially when the video domain changes. The gap between in-domain prediction and cross-domain transfer remains the main limitation.

A qualitative nearest-neighbor view of the predicted states is available for the [human model](docs/figures/imagined-phone-h2.png) and [robot model](docs/figures/imagined-robots-h2.png). These examples illustrate model behavior; they are not an additional quantitative evaluation.

## Data

| | Videos (train / val / test) | Content |
|---|---|---|
| Human ([h2r-phone](https://huggingface.co/datasets/batgre/h2r-phone)) | 124 / 14 / 20, 20.5 min | a person handling objects, filmed with a phone |
| Robots ([h2r-robots-1h30](https://huggingface.co/datasets/batgre/h2r-robots-1h30)) | 173 of 712 used for training | real robot videos from 6 Open-X datasets (LeRobot): DROID, Berkeley UR5, Jaco Play, TACO Play, Stanford Kuka, RoboTurk |
| Simulation ([h2r-sim-v3](https://huggingface.co/datasets/batgre/h2r-sim-v3)) | 1,100 scripted + 1,100 random episodes, split 70 / 10 / 20 (61,680 / 8,400 transitions for the controllers) | robosuite Panda, Lift, `agentview` camera, 40 actions per episode, with the actions |

The world models only see the human or the robot videos. The simulation is only used for the controllers and the
evaluation. All videos are sampled at 4 fps and 224 px and split by video.

Phone videos were recorded from two camera angles: the first 81
videos (65 / 6 / 10) from one angle, the next 77 (59 / 8 / 10) from another. Location metadata and audio are removed before upload.

Trained model: the world model learned from human videos (run 2) and its controller, in
[batgre/h2r-lift](https://huggingface.co/batgre/h2r-lift) ([world model](https://huggingface.co/batgre/h2r-lift/tree/main/wm-phone-h2),
[controller](https://huggingface.co/batgre/h2r-lift/tree/main/ctrl-phone-h2)).

## Downstream control

We freeze each world model and train a small controller on Robosuite transitions to predict the IDM latent action from the robot's measured end-effector action and current visual state. CEM then searches over candidate robot actions, rolls them out through the controller and world model, executes the first action, and replans after observing the result.

This is an experimental test of whether learned visual dynamics are usable for action selection. In both runs, **neither planner lifted the cube successfully in 30 held-out Robosuite Lift episodes**. The demonstration-replay baseline succeeds in 73–87% of trials across three repeated evaluations, while random control succeeds in none. The planner receives pre-grasp, grasp, and lift goal frames extracted from the test demonstration; these subgoals are supplied, not discovered.

Thus the current results support limited in-domain learning of action-conditioned visual dynamics, **not successful human-to-robot transfer**.

## Reproduce

`lift.ipynb` runs every step below on Colab, in order.

```bash
uv sync
uv run hf auth login && uv run wandb login     # ffmpeg is also needed (brew install ffmpeg on macOS)
```

**Optional: run the trained planner** on 5 test states:

```bash
uv run hf download batgre/h2r-lift --local-dir outputs --include "wm-phone-h2/*" --include "ctrl-phone-h2/*"
uv run hf download batgre/h2r-sim-v3 --repo-type dataset --local-dir data/h2r-sim --include "raw/Panda/Lift/scripted/*"
uv run h2r eval pick --set name=try controller=outputs/ctrl-phone-h2 "policies=[planner]" episodes=5 planner.horizon=2
```

Results are saved in `outputs/try/results.json`. If the policy produces an unsuccessful episode, its frames are saved as `episode_failure.png`; `episode_success.png` is produced only when a successful episode is observed.

**Train again** (data, world models, controllers, pick):

```bash
uv run hf download batgre/h2r-phone --repo-type dataset --revision 815bfb769aa6fc2059fe746381473915a2fb665d --local-dir data/h2r-phone --exclude "raw/*"
uv run hf download batgre/h2r-robots-1h30 --repo-type dataset --local-dir data/h2r-robots-1h30
uv run hf download batgre/h2r-sim-v3 --repo-type dataset --local-dir data/h2r-sim

H="rollout_steps.end=2 val_horizon=2 iterations=1000 eval_every=50"
uv run h2r train --set name=wm-phone-h2 data.root=data/h2r-phone/ data.hf_repo=batgre/h2r-phone $H
uv run h2r train --set name=wm-robots-h2 data.root=data/h2r-robots-1h30/ data.hf_repo=batgre/h2r-robots-1h30 data.max_videos=173 $H
uv run h2r controller train --set name=ctrl-phone-h2 world_model=outputs/wm-phone-h2 iterations=4000 eval_every=100
uv run h2r controller train --set name=ctrl-robots-h2 world_model=outputs/wm-robots-h2 iterations=4000 eval_every=100

uv run h2r eval pick --set name=pick-baselines-v3 "policies=[random, replay]" episodes=30
uv run h2r eval pick --set name=pick-phone-h2 "policies=[planner]" controller=outputs/ctrl-phone-h2 episodes=30 planner.horizon=2
uv run h2r eval prediction --set controller=outputs/ctrl-phone-h2 horizon=2        # also counterfactual, ranking, planning, interface
```

Every command takes `--config` and `--set key=value ...` (nested keys with dots). A run goes to `outputs/<name>`;
running a training again continues it from `last.pt`, and an evaluation that already wrote `results.json` is skipped.
`h2r data extract`, `h2r data collect` and `h2r data push` rebuild the datasets (`configs/data/`). Tests:
`uv run pytest`.

## Experimental setup and limitations

- **Matched world-model budget.** Run 2 uses the same architecture, training steps and approximately the same number of sampled frames for human and robot videos, but not identical motion statistics, video lengths or numbers of clips.
- **Separate training stages.** The simulation is used to fit the downstream action controller and to evaluate control, not to train either video-only world model.
- **Oracle-action prediction metrics.** Validation rollouts use IDM actions inferred from observed future pairs. A high score does not imply that the system can select correct future actions.
- **Limited statistical evidence.** One training seed per world model, a small human-video dataset from one person, and 30 held-out simulation episodes.
- **Not an ablation.** Run 1 and Run 2 vary the data amount, viewpoints and prediction horizon together.
- **Domain shift.** Neither world model predicts Robosuite dynamics better than persistence; the downstream pick policies fail.

## References and acknowledgments

`src/h2r/models/jepa.py` and `src/h2r/models/module.py` are adapted from
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Copyright (c) 2026 Lucas Maes), released under the MIT
license, in [`LICENSES/LeWM-MIT.txt`](LICENSES/LeWM-MIT.txt). The IDM, the controller and the training values follow
Garrido et al. 2026.

- Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero. _LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels._ arXiv:2603.19312, 2026.
- Quentin Garrido, Tushar Nagarajan, Basile Terver, Nicolas Ballas, Yann LeCun, Michael Rabbat. _Learning Latent
  Action World Models In The Wild._ arXiv:2601.05230, 2026.
- Basile Terver et al. _What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?_
  TMLR, 2026.
- Maxime Oquab et al. _DINOv2: Learning Robust Visual Features without Supervision._ TMLR, 2024.
- Open X-Embodiment Collaboration. _Open X-Embodiment: Robotic Learning Datasets and RT-X Models._ 2023.
- Yuke Zhu et al. _robosuite: A Modular Simulation Framework and Benchmark for Robot Learning._ 2020.
