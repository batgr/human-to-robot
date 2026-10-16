# Human-to-Robot

Can a robot learn to act from videos of people? A world model learns from video alone how a scene changes: an inverse
dynamics model (IDM) infers a latent action between two frames, and a predictor rolls the scene forward with it. A
small controller then maps the robot's real actions to these latent actions, so that a planner can search for the
robot actions whose predicted outcome reaches a goal image. The measure is pick success in closed loop, in simulation
(robosuite, Panda, Lift).

Question: does a world model learned from human videos serve a robot as well as one learned from robot videos? Two
world models, same code, compute and number of training frames, only the video differs: human videos filmed with a
phone, and real robot videos. Each gets a controller trained on the same simulated robot episodes.

## Results

### World models

_To be added: training curves (prediction loss, SIGReg, KL), skill(h), controllability, effective rank, active latent
dimensions, straightness, phone vs robot videos, each with a short analysis._

### Controllers

_To be added: training and validation loss, validation R², for each world model._

### Pick success

| Policy | Success (95% interval) | Reached | Grasped |
|---|---|---|---|
| random | | | |
| replay | | | |
| planner, phone world model | | | |
| planner, robot world model | | | |

## Data and models

- Human videos: https://huggingface.co/datasets/batgre/h2r-phone
- Robot videos: https://huggingface.co/datasets/batgre/h2r-robots-1h30
- Simulated robot episodes: https://huggingface.co/datasets/batgre/h2r-sim-v3
- World models and controllers: https://huggingface.co/batgre/h2r-lift

## Try the trained model

```bash
uv sync
uv run hf download batgre/h2r-lift --local-dir outputs
uv run hf download batgre/h2r-sim-v3 --repo-type dataset --local-dir data/h2r-sim --include "raw/Panda/Lift/scripted/*"
uv run h2r eval pick --set name=try controller=outputs/ctrl-phone "policies=[planner]" episodes=5
```

The planner, with the world model learned from phone videos, picks the cube from 5 test states (`ctrl-robots` for
the one learned from robot videos). Results in `outputs/try/results.json`, frames of a success and of a
failure in `outputs/try/episode_success.png` and `episode_failure.png`.

## How it works

| Component | Input -> output | Trained on |
|---|---|---|
| Encoder | frame -> tokens | frozen (DINOv2-small) |
| IDM | two consecutive states -> latent action | video |
| Predictor | past states and latent actions -> next state | video |
| Controller | real action and current state -> latent action | robot episodes with actions |
| Planner | current frames, goal frame -> next action | none (CEM over real actions, replans after every observation) |

## Training recipe

| | World models | Controllers |
|---|---|---|
| Data | phone: 65 videos, 2,580 frames; robots: 102 videos, 2,587 frames (4 fps, 224 px) | 1,542 robosuite episodes, 61,700 transitions (Panda, Lift, scripted and random) |
| Trained | projector (DINOv2-small patches -> 4 x 4 x 128), predictor (6 layers), IDM (2 layers) | 2 attention layers on the current state + action embedding |
| Objective | prediction MSE + SIGReg (0.09) + KL of the latent action (beta 1e-3) | L2 to the IDM's latent action |
| Unrolled | context 3 frames; rollout K from 1 to 5 over the first half (scheduled sampling) | one step |
| Optimizer | Muon 0.02 (matrices) + AdamW 2.2e-4, weight decay 0.04, warmup 10%, cosine, clip 1.0 | AdamW 1e-3, weight decay 0.04, warmup 300, cosine |
| Batch, steps | 128 clips of 8 frames, 1,000 steps (about 60 passes) | 256 transitions, 4,000 steps (about 17 passes) |
| Kept | best validation skill, checked every 50 steps | best validation loss, checked every 100 steps |

The encoder is frozen. Both world models get the same code, steps and number of training frames, so that only the video
differs. With about 2,000 clips the world models see each one many times: the predictor's dropout (0.1), the weight
decay and the checkpoint chosen on validation guard against overfitting. The controllers have 30 times more transitions
than in the earlier small-budget runs, which overfit after 250 steps.

Planner: CEM over Δxyz and the gripper (rotations at 0), 300 samples, 10 elites, 10 iterations, horizon 5, the first
action executed and the plan made again after every observation; goal frames: pre-grasp, grasp and lift.

## Usage

### Setup

```bash
uv sync
uv run hf auth login
uv run wandb login
```

`ffmpeg` must be installed (`brew install ffmpeg` on macOS).

### Data

Robot episodes (robosuite, `configs/data/sim.yaml`), downloaded or collected again:

```bash
uv run hf download batgre/h2r-sim-v3 --repo-type dataset --local-dir data/h2r-sim
uv run h2r data collect --config configs/data/sim.yaml   # <robot>/<task>/<policy>/episode_<seed>/: videos, state.parquet, sim.parquet
uv run h2r data extract --config configs/data/sim.yaml   # frame shards; actions: end-effector deltas
uv run h2r data push --config configs/data/sim.yaml
```

Any folder of videos (`raw_dir` in `configs/data/config.yaml`):

```bash
uv run h2r data extract   # 4 fps, 224 px, split by video
uv run h2r data push
```

### Training

```bash
uv run h2r train --set name=wm-phone data.root=data/h2r-phone/ data.hf_repo=batgre/h2r-phone
uv run h2r train --set name=wm-robots data.root=data/h2r-robots-1h30/ data.hf_repo=batgre/h2r-robots-1h30 data.max_videos=102
uv run h2r controller train --set name=ctrl-phone world_model=outputs/wm-phone
```

Every command takes `--config` and `--set key=value ...` (nested keys with dots). A run goes to `output_dir/name`:
running a training again continues it from `last.pt`, and an evaluation that already wrote `results.json` is not run
again. `data.max_videos` sets a data budget, the same episodes for the world model and the controller; `init_from`
fine-tunes a trained world model. Each run writes `metrics.jsonl` and, at the end, `curves.png`.

### Evaluation

```bash
uv run h2r eval pick --set controller=outputs/ctrl-phone             # random, replay and planner on the same test states
uv run h2r eval interface                                           # requested vs obtained displacement of the commands
uv run h2r eval prediction --set controller=outputs/ctrl-phone       # skill(h): IDM (oracle), controller, controller in the loop
uv run h2r eval counterfactual --set controller=outputs/ctrl-phone   # sequences from one state: different and coherent outcomes?
uv run h2r eval ranking --set controller=outputs/ctrl-phone          # planner cost vs real outcome of played sequences
uv run h2r eval transfer                                            # human-to-robot transfer from pick runs per arm
```

A pick succeeds when the object is lifted 4 cm while grasped.

### Tests

```bash
uv run pytest
```

## Credits

`src/h2r/models/jepa.py` and `src/h2r/models/module.py` are adapted from
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Copyright (c) 2026 Lucas Maes),
released under the MIT license. The license text is in [`LICENSES/LeWM-MIT.txt`](LICENSES/LeWM-MIT.txt).

Paper:

> Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero.
> _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels._
> arXiv:2603.19312, 2026.
