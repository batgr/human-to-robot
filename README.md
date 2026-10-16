# Human-to-Robot

## Setup

```bash
uv sync
```

`ffmpeg` must be installed (`brew install ffmpeg` on macOS).

## Data

2. Extract the frames:

   ```bash
   uv run h2r data extract
   ```

3. Push to the Hugging Face Hub (after `uv run hf auth login`):

   ```bash
   uv run h2r data push
   ```

Simulation episodes with actions (robosuite), written to `raw/` like any video folder:

```bash
uv run h2r data collect --config configs/data/sim.yaml   # <robot>/<task>/<policy>/episode_<seed>/: videos, state.parquet, sim.parquet
uv run h2r data extract --config configs/data/sim.yaml   # actions: end-effector deltas, as for DROID
```

## Training

```bash
uv run wandb login
uv run h2r train --config configs/training/world_model.yaml
uv run h2r train --set name=wm-A data.root=data/h2r-sim/ data.max_videos=10   # any config value, nested with dots
```

Every command takes `--set key=value ...`. A run goes to `output_dir/name`: running a training again continues it from
`last.pt` (with its own config), and an evaluation that already wrote `results.json` is not run again.

Training runs a fixed number of iterations, whatever the data budget. The rollout depth K grows during training
(`rollout_steps`: teacher forcing first, then up to the planning horizon). Validation, every `eval_every` iterations: the
losses, skill(h) = 1 − error / error of persistence for h = 1..5 (predictor unrolled with
the IDM's latent actions), controllability, effective rank of the states and active latent dimensions. `best.pt` has
the best mean skill.

Controller (real action + current frame -> latent action) for a trained world model, on episodes with actions:

```bash
uv run h2r controller train --set world_model=outputs/h2r
```

Validation: L2 to the IDM's latent action and R² against always predicting the mean one. `init_from` starts from a
trained controller (e.g. Panda -> Sawyer).

Data budget (world model and controller): `data.include` and `data.max_videos` keep the same episodes for both, whatever
the training seed, and a smaller budget is part of a larger one. `init_from` in the world model config fine-tunes a
trained world model (e.g. SSv2, then robot videos).

## Evaluation

The central measure is pick success in closed loop: on held-out test episodes, the planner replans after every
observation; a pick succeeds when the object is lifted 4 cm while grasped. Baselines: random commands, and a replay of
the demonstration's actions (a check of the command interface).

```bash
uv run h2r eval pick --config configs/eval/pick.yaml             # random, replay and planner on the same initial states
uv run h2r eval transfer --config configs/eval/transfer.yaml     # arms A (robot only), B (SSv2 + robot), C (SSv2 frozen)
```

`h2r eval transfer` reads pick results per (arm, robot episodes, seed): success per budget, gain over the robot-only arm
with a paired bootstrap over the initial states and the seeds, and the budgets between which the robot-only arm reaches
the same success.

Diagnostics, to explain a success or a failure (same episodes and planner, `configs/eval/diagnostics.yaml`):

```bash
uv run h2r eval interface        # requested vs obtained displacement of the commands
uv run h2r eval prediction       # skill(h) with the IDM (oracle), the controller, the controller in the loop
uv run h2r eval counterfactual   # sequences played from one state: are their predicted outcomes different and coherent?
uv run h2r eval ranking          # planner cost vs real outcome (gripper and object) of played sequences
```

## Tests

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
