# Human-to-Robot

```bash
uv sync

# raw videos -> frame shards (configs/data/config.yaml)
uv run h2r data extract --set raw_dir=data/raw/ processed_dir=data/processed/

# frame shards -> Hugging Face Hub
uv run h2r data push --set processed_dir=data/processed/ hf_repo=<user>/<dataset>

# DROID episodes for the controller (configs/data/droid.yaml), then extract / push with the same config
uv run h2r data droid
uv run h2r data extract --config configs/data/droid.yaml
uv run h2r data push --config configs/data/droid.yaml

# controller for a trained world model; ablations: --set 'model.inputs=[state]' or 'model.inputs=[action]'
uv run h2r train controller --set world_model=outputs/<wm-run>

# closed loop in robosuite (configs/planning/robosuite.yaml, environment configs/env/config.yaml, CEM
# configs/planning/cem.yaml): policy random | scripted | recorded | zero | cem, CEM goal final | stages | dense
# (frame t + k); stage rates, summary.json and one video per episode in outputs/planning/
uv run h2r plan run --set policy.goal=dense policy.k=1 policy.horizon=1 policy.context=2
uv run h2r plan run --set policy=configs/planning/<another cem>.yaml   # another CEM file
uv run h2r plan run --set policy.type=recorded                          # a baseline
# same, watched in the MuJoCo window (macOS needs mjpython)
uv run mjpython -m h2r.cli plan run --set env.has_renderer=true episodes.count=1

# tests
uv run pytest
```

## Current model

The reference world model is the 20 h run of experiment 12: `wm-cls-ssv2-20hall-k2from1050-b3e-4-20k`, checkpoint `step-20000.pt`, trained at commit `248588a`.

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
| Stop-gradient on the target | None |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-3, 10% warmup, gradient clip 1.0, bf16 |
| Length | 20,000 steps, batch 128, seed 0 |

Measured on validation at step 20,000:

| Metric | Value |
|---|---|
| Error / persistence | 0.0135 |
| 7-step gain over persistence | 0.987 |
| Controllability | 0.99 |
| Leak | 0.015 |
| Invented motion | 1.09 |
| Rank of the states (RankMe) | 72 / 192 |
| Active dimensions of z | 8 / 64 |
| Action class from z | 0.015 (chance 0.006) |
| Left / right from z | 0.50 (chance 0.50) |
| Hand R² from the encoder | 0.08 |

These numbers do not show that the model is usable for control: see experiments 14 to 16.

### Planning metrics of the current model

Kept up to date: every new world model gets the same measures, on the same windows, so its progress on the task can be followed. Robosuite Lift, simulation validation episodes (210), controller `ctrl-sim`, context of 2 states, 3 actions planned. Intervals are 95 % bootstrap over episodes.

| Metric | Value | Reference |
|---|---|---|
| Success rate: reach / grasp / lift (closed loop) | best lift 1 / 20, see the closed-loop table below | recorded actions 16 / 20 |
| Offline verdict, run once: Δxyz L1 error of the CEM plan, Sim test (ctrl-sim) / DROID test (ctrl-droid) | 0.126 [0.121, 0.131] / 0.403 [0.391, 0.415]: fails (rule: beat the zero plan overall and on the most moving third) | zero plan 0.092 / 0.087, another episode 0.136 / 0.135 |
| Hit rate Top-1 / Top-5 of the true actions among 64, goal 3 steps ahead | 10.4 % [8.9, 12.0] / 31.9 % [29.2, 34.6] | chance 1.6 % / 7.8 % |
| Same, goal 7 steps ahead (true actions are no longer the best plan) | 4.5 % [3.2, 5.8] / 20.1 % [17.7, 22.6] | chance 1.6 % / 7.8 % |
| Δxyz L1 error of the CEM plan (offline FDE), real goal 7 steps ahead | 0.122 [0.114, 0.129] | zero plan 0.087 [0.081, 0.093], another episode's move 0.130 [0.125, 0.135] |
| Same Δxyz L1 error by policy: scripted / random episodes | 0.086 / 0.169 | zero plan 0.061 / 0.121; CEM minus zero plan on the most moving scripted third −0.001 [−0.013, +0.010] |
| Direction of the CEM plan (cosine Δxyz), real goal 7 steps ahead | 0.16 [0.13, 0.20], amplitude 1.33 × the true move | random plan 0.00 |
| True actions cost less than another episode's, goal 3 steps ahead | 0.745 [0.734, 0.757] | chance 0.5, pass rule 0.75 |
| True actions bring the state closer than a zero plan, goal 7 steps ahead | 0.633 [0.605, 0.662] | chance 0.5 |
| Direction of the planned move (cosine Δxyz), goal imagined from the true actions | 0.42 [0.28, 0.56] | random plan 0.00 |
| Same goal imagined 7 steps ahead (no model error on the goal): Δxyz L1 error / cosine to the 3 / 7 true steps | 0.114 [0.107, 0.120] / 0.17 / 0.30 | real goal 0.122 / 0.16 / 0.30; zero plan 0.087; 3 true steps 0.62 to the 7-step move |
| Same goal imagined 7 steps ahead, all 7 actions planned (diagnostic): cosine to the true 7-step move / final elite cost | 0.31 [0.27, 0.34] / 0.00047 | 3 actions planned 0.30; true actions cost 0 |
| Δxyz L1 error of the CEM plan, 1 action, goal 1 step ahead | 0.044 [0.041, 0.046]; most moving third: gain on the zero plan +0.0045 [+0.0020, +0.0070] | zero plan 0.038 [0.035, 0.041], another episode 0.061 |
| Gripper closure readable from the state (linear probe R²) | 0.04 | position x 0.81, z 0.58; time alone 0.05 |

#### Closed loop (MPC in robosuite)

20 scripted simulation validation episodes whose recording lifts the cube, same seed (same cube) as the recording. At each step the rendered frame goes through the data chain (H.264, then JPEG) and the encoder. The CEM (300 plans, 30 elites, 10 iterations, start at the controller's action mean and std) plans H actions toward the goal. The first action is executed by a calibrated delta command (command = Δxyz / K, held 5 control steps; rotation 0; gripper closes or opens past a Δclosure of ±0.05). Then it replans: 40 steps, context of 2 states linked by the controller's z of the achieved action.

Stages, each counted if met at any step:
- **reach**: gripper centre within 3 cm of the cube centre;
- **grasp**: both finger pads touch the cube (robosuite `_check_grasp`);
- **lift**: cube centre more than 4 cm above the table (robosuite Lift success test).

Goals:
- **final goal**: the last frame of the recording (cube held up), at every step;
- **stage subgoals** (as V-JEPA 2-AC's subgoal images): the recorded frame where reach is first met, then the one where grasp is first met, then the final frame. Each one is the goal until the recording reaches that stage (stage times from an exact replay of the recorded commands);
- **dense subgoals t+k**: the recorded frame k steps after the current step.

| Planner | Reach | Grasp | Lift | Steps to reach / grasp / lift | Final error: gripper / cube | Hit rate Top-1 / Top-5 (offline) | Δxyz FDE (offline) | Time per step |
|---|---|---|---|---|---|---|---|---|
| Recorded actions, same executor (reference) | 18 [70–97 %] | 19 [76–99 %] | 16 [58–92 %] | 10 / 17 / 18 | 1.7 / 1.7 cm | | | 0.43 s |
| Final goal, H = 1 | 0 [0–16 %] | 0 [0–16 %] | 0 [0–16 %] | – / – / – | 6.7 / 11.9 cm | not measured | not measured | 0.56 s |
| Final goal, H = 2 | 1 [1–24 %] | 1 [1–24 %] | 0 [0–16 %] | 22 / 10 / – | 14.9 / 11.9 cm | not measured | not measured | 0.70 s |
| Final goal, H = 3 | 0 [0–16 %] | 0 [0–16 %] | 0 [0–16 %] | – / – / – | 19.9 / 11.9 cm | not measured | not measured | 0.87 s |
| Stage subgoals, H = 1 | 3 [5–36 %] | 1 [1–24 %] | 0 [0–16 %] | 20 / 11 / – | 14.8 / 12.1 cm | not measured | not measured | 0.64 s |
| Stage subgoals, H = 2 | 5 [11–47 %] | 3 [5–36 %] | 1* [1–24 %] | 14 / 8 / 24 | 18.5 / 12.1 cm | not measured | not measured | 0.76 s |
| Dense subgoals t+1, H = 1 | 4 [8–42 %] | 3 [5–36 %] | 1 [1–24 %] | 17 / 13 / 23 | 6.9 / 11.9 cm | not measured | 0.044 (zero plan 0.038) | 0.47 s |
| Dense subgoals t+7, H = 3 | 7 [18–57 %] | 3 [5–36 %] | 0 [0–16 %] | 13 / 16 / – | 19.5 / 12.0 cm | 4.5 % / 20.1 % (chance 1.6 / 7.8) | 0.122 (zero plan 0.087) | 0.86 s |
| Zero plan (reference) | 0 [0–16 %] | 0 [0–16 %] | 0 [0–16 %] | – / – / – | 12.4 / 11.9 cm | | | 0.43 s |

Counts out of 20, 95 % Wilson intervals. Steps to a stage: median over the episodes that reach it. Final error: distance at step 40 to the recording's final gripper position and final cube position. Offline columns: same horizon and goal distance on Sim val windows. Time per step: wall clock on a MacBook (MPS), rendering and simulation included (0.43 s without planning).

- **Measured:** no planner lifts the cube in a stable way. The one lift with dense subgoals t+1 (episode 20) raises the cube 9 cm, grasped at step 23, and drops it 2 steps later. *The stage-subgoal lift (episode 134) is the cube at 4.1 cm for one step, without a grasp at that step: pushed, not lifted (inferred).
- **Measured:** grasps last 1 or 2 steps (recorded actions: 22). The gripper is commanded closed / open 6 to 8 times per episode (recording: once).
- **Measured:** with the final goal alone the arm moves about 8 cm in 5 steps, then stays 13–14 cm from the cube, for every H. With subgoals it comes to about 7 cm near step 15, then moves away when the goal becomes the lifted cube. Every planner ends 13–16 cm from the cube.
- **Measured:** the recorded actions through the executor lift 16 / 20 (recording 20 / 20 on these episodes): without replanning, a drift of 1–2 cm at closing time flips marginal grasps (on 119 episodes: 10 lost, 9 gained).
- **Inferred:** the subgoals help the approach (reach 3–7 / 20 against 0–1 with the final goal), not the grasp: the state does not hold the gripper closure (probe R² 0.04), so the CEM plans it at random.

```bash
uv run h2r train wm --set name=wm-cls-ssv2-20hall-k2from1050-b3e-4-20k data.processed_dir=<dir> \
    loss.rollout_steps=2 loss.rollout_from=1050 loss.beta=0.0003 iterations=20000
```

## Experiments

The model is a JEPA world model (LeWM recipe): a ViT-tiny encoder trained from scratch, an IDM that infers a latent action z from two consecutive states, and a predictor conditioned on z. It is trained on Something-Something v2 (SSv2) video at 4 fps, with no action labels. Each experiment below answers one question. Every result is labelled **measured** (observed in a run) or **inferred** (our reading of it).

Metrics used throughout, all on held-out validation videos:
- **error / persistence**: next-state prediction error divided by the error of copying the current state. Below 1 means the model beats copying.
- **controllability**: how much the prediction changes when z comes from another clip (0 means z is ignored).
- **leak**: how much of the future a z carries when transplanted into another clip (copying the future instead of describing an action).
- **invented motion**: on static clips, predicted change / true change (1 is ideal).

Unless stated otherwise: batch 128, lr 1e-4, KL weight β 1e-4, SIGReg λ 0.09, one seed.

Data, split by whole video or episode:

| Dataset | Content | Used for |
|---|---|---|
| SSv2 10 h | 20 action classes | World model |
| SSv2 20 h | 19,301 videos, 168 action classes (classes where the camera moves are left out) | World model |
| [DROID](https://huggingface.co/datasets/batgre/h2r-droid) | 10 h of real robot episodes, left exterior camera, with actions | World model control, controller |
| [Robosuite](https://huggingface.co/datasets/batgre/h2r-sim-v3) | Panda Lift, 1,100 scripted and 1,100 random episodes, with actions | Controller |

### 1. Does the latent action carry the prediction?

1,000 steps on 5 h of SSv2, the normal model against z = 0.

- **Measured:** error / persistence is 0.82 with z and 1.10 with z = 0. Without z the model never beats copying.
- **Measured:** the temporal and posterior collapses seen when overfitting one batch do not appear on real data.
- **Inferred:** the IDM carries the gain. The overfit test cannot judge z, because the predictor memorises a single batch from context alone.

### 2. Learning rate, predictor form and seed noise

1,000 steps each.

| Question | Result (error / persistence) |
|---|---|
| Learning rate 2.5e-5 / 5e-5 / 1e-4 / 2e-4 / 4e-4 | 0.92 / 0.82 / 0.48 / 0.38 / 0.57; 4e-4 is unstable (z dimensions oscillate) |
| Direct vs residual predictor (s_t + Δ, zero-initialised) | 0.475 vs 0.787. The residual predictor plateaus from step 200 and predicts changes half their true size |
| Seed noise, 3 seeds of the base | 0.40 / 0.27 / 0.35 (std 0.05) |

- **Measured:** spread across seeds follows how much z is used. The IDM switches on at a seed-dependent step (200–400).
- **Inferred:** single-seed gaps under about 0.1 are noise.
- **An invalid experiment, caught:** a first CLS-vs-patch comparison reported 0.48 vs 0.77. Both runs were in fact CLS, because the Colab code ignored a config key. The 0.3 gap came from initialisation alone. Since then, runs assert the state shape.

### 3. KL weight and SIGReg weight

1,000 steps.

| | error / persistence | KL (nats) | Notes |
|---|---|---|---|
| β 0 / 1e-4 / 1e-3 / 1e-2 | 0.35 / 0.40 / 0.75 / 0.99 | 158 / 34 / 3.2 / 0.23 | content leak 2.2 % / 0.8 % / 0.3 %; β 1e-4 kept |
| λ 0.03 / 0.09 / 0.27 | 0.28 / 0.40 / 0.41 | | effective rank 36 / 54 / 63; gradient clipping on 11 % / 52 % / 100 % of steps |

- **Measured:** a larger β halves the predicted motion (0.83 → 0.34 of the true change on moving clips). This is the KL bottleneck.
- **Measured:** invented motion on static clips is the same at every β (1.6–1.8). It comes from the predictor, not from z.

### 4. Does more video help?

Nested SSv2 subsets, 3,000 steps each.

![Validation error against the amount of data](docs/figures/experiments/data-scaling.png)

- **Measured:** error / persistence falls by about 20 % per doubling of data (0.29 at 1 h → 0.14 at 8 h), with no sign of saturation.
- **Measured:** at equal hours, 20 classes beat 10 classes (0.116 vs 0.143), even on the original classes.
- **Inferred:** diversity matters more than quantity at this scale.

### 5. Does a 2-step rollout loss help?

3 seeds per condition, 1,000 steps.

![Gain over persistence and leak for each rollout loss variant](docs/figures/experiments/rollout-leak.png)

- **Measured:** with K = 2, every prediction metric improves (7-step gain over persistence, 1 - error / persistence: 0.68 → 0.96). The leak rises from 1 % to 66 %.
- **Inferred:** the predictor reads its own noisy prediction while z comes from an IDM that sees the true future. Putting the future into z is the cheapest way down the loss. These gains rely on that leak, which is unavailable at planning time.
- **Measured:** two fixes bring the leak back to about 3 %: starting K = 2 at step 500 (E9), or applying the term to the predictor only (E10).
- **How we use it:** leak is now a hard constraint before ranking any model on prediction metrics. Experiment 11 shows that this 3 % overstates the fix.

### 6. Long training, and the positive control on robot video

20,000 steps, one run per dataset: SSv2 10 h / 20 classes, and DROID 10 h (robot video, left exterior camera).

![Long runs on human (SSv2) and robot (DROID) video](docs/figures/experiments/long-run-ssv2-droid.png)

On SSv2 (measured):
- error / persistence reaches 0.054 and 7-step gain over persistence 0.92;
- controllability is 0.97, and the leak stays at 0.65 % for the whole run;
- invented motion falls to 1.05;
- validation error is still falling when the learning rate reaches 0.

On DROID (measured):
- error / persistence plateaus at 0.78;
- the IDM wakes up only around step 2,500 and uses 2 nats;
- a controller that maps the real robot actions into z carries no action information. Its prediction error is the same as with another clip's actions, and the probe from z to the action gives R² 0.0004.

**Inferred:** the recipe fails on robot video too, so the failure is in the representation, not in the human-to-robot gap.

### 7. Do the states know where the hand is?

Linear probes, held out by video, on SSv2 validation frames with a hand tracked by MediaPipe.

![Hand position decodable from each representation](docs/figures/experiments/hand-probe.png)

![Probe per encoder block](docs/figures/experiments/hand-probe-layers.png)

- **Measured:** our world-model states give R² 0.00, and our encoder's patch tokens 0.07, close to an untrained encoder (0.04). Frozen DINOv2 patches give 0.39–0.43.
- **Measured:** layer by layer, our encoder stays flat at 0.04–0.09 through all 12 blocks. The information is never built, rather than built and then lost.
- **Measured:** with a frozen DINOv2 encoder, our patch projector and loss keep the hand (0.22). A CLS state on the same encoder loses it (0.01).
- **Inferred:** the main cause is the encoder trained from scratch, not the objective erasing the information. Pooling to one CLS token adds a second loss.

### 8. Which loss term trains the encoder?

Share of each term in the encoder gradient, ⟨g_term, g⟩ / |g|², on one batch.

![Share of each gradient path in each encoder layer](docs/figures/experiments/gradient-paths.png)

- **Measured:** SIGReg supplies about 90 % of the encoder gradient at every layer.
- **Measured:** the prediction loss reaches the encoder through two paths that cancel out (cosine −0.91 to −0.97):
  - the context path, through the predictor's input;
  - the target path, which pulls the target toward the prediction.
- **Measured:** the net prediction share is about 8 %.

### 9. Does the scene alone satisfy SIGReg?

SIGReg tests whether a set of states looks Gaussian (a Gaussian set scores 1.05).

| Set of 12 states | SIGReg statistic |
|---|---|
| frames from 12 different videos (as in a training batch) | 1.08–1.28 |
| 12 frames of one video | 9.6–10.1 |

- **Measured:** 79–86 % of the state variance lies between videos, for every network including an untrained one (84 %), and at every λ.
- **Inferred:** a batch of mixed videos satisfies SIGReg through scene differences alone, so SIGReg does not push the encoder to represent what changes inside a video.

### 10. Does a stop-gradient on the target help?

Patch-16 states (a 4 × 4 grid), stop-gradient on the target as in AD-E2E-JEPA, compared with the long CLS run (same schedule). The run stopped near step 9,500 and is shown up to step 6,500.

![Stop-gradient run against the CLS run](docs/figures/experiments/stop-gradient.png)

- **Measured:** error / persistence stalls at 0.77 from step 1,500, then rises to 0.85. Controllability stays at 0.10–0.16. States change 2 to 5 times more per frame than in the CLS run. Hand R² stays at 0.02–0.04.

To find the cause, we measured the checkpoints at steps 1,500, 2,000 and 4,000 on 480 validation clips:

![Diagnostic of the stop-gradient run](docs/figures/experiments/stop-gradient-diagnostic.png)

- **Measured:** the IDM's readout attends to 2–3 fixed cells of the grid, unrelated to where pixels change. The run without stop-gradient does the same, yet uses z well (gain 0.69). This is not the cause.
- **Measured:** z is unrelated to the change of state. The linear R² of z from Δs is 0.00, against 0.82 without stop-gradient and 0.86 for CLS. z is mostly a constant that the predictor uses as a bias.
- **Measured:** the states stay close to pixels. A 2-pixel shift moves them by 40–57 % of a real time step, against 20–25 % for the trained baselines and 69 % for an untrained encoder.
- **Inferred:** the target path is what makes an encoder trained from scratch predictable. AD-E2E-JEPA can use a stop-gradient because its encoder is a frozen DINOv3.
- **Untested:** a stop-gradient with a frozen encoder, an EMA target, or a partial target gradient.

### 11. How much does the leak metric hide?

Audit of `leak` on the saved checkpoints of experiments 5 and 6: 1,024 validation clips, means over 3 seeds (one long run). A is the clip that gives z, B the clip that receives it, C an unrelated clip.

| Measure | K = 1 | K = 2 | K = 2 from step 500 | Predictor only | Long K = 1 |
|---|---|---|---|---|---|
| leak | 0.011 | 0.662 | 0.034 | 0.031 | 0.007 |
| prediction closer to A than to C (0.5 = nothing of A) | 0.572 | 0.853 | 0.678 | 0.691 | 0.503 |
| part of the move that belongs to A (α_A − α_C) | 0.040 | 0.252 | 0.090 | 0.093 | 0.003 |

- **Measured:** `leak` only fires past halfway from B to A, so it is blind to everything below. The fix looks like a 95 % reduction by `leak`, and 64 % by α_A − α_C.
- **Measured:** the long K = 1 run has no content specific to A (interval 0.479–0.527 around 0.5).
- **How we use it:** both threshold-free measures now run at every validation (`latent_action/leak_a_vs_c`, `leak_alpha`).

### 12. Long training on 20 h of all classes

20,000 steps on SSv2 20 h, K = 2 from step 1,050, β 3e-4.

Measured at the end of the run:
- error / persistence 0.0135, 7-step gain over persistence 0.987, controllability 0.99;
- leak 0.015, but prediction closer to A than to C 0.572, so some content of A remains;
- hand R² from the encoder 0.08;
- action class from z 0.015 (chance 0.006).

**Untested:** the gap with experiment 6 could come from K, β or the data. Only a K = 1 run on the same data and β can tell.

### 13. Do the states separate datasets or actions?

4,148 clips from 11 sources for the datasets, 5,040 SSv2 clips for the classes. Share of the 10 nearest clips with the same label.

| Space | Same source (chance about 0.14) | Same action class (chance 0.006) |
|---|---|---|
| Pixels | 0.75 | 0.013 |
| Untrained encoder | 0.79 | 0.012 |
| Trained encoders | 0.81–0.85 | 0.013–0.017 |

- **Measured:** an untrained network already separates the sources, so most of this separation is not learned.
- **Measured:** the states do not group by action class. The model trained on DROID alone matches the SSv2 models on SSv2 classes.

### 14. Do real robot actions drive the model?

One controller per world model and dataset, 100,000 steps. Seven open-loop steps on test episodes, 100 pairs of clips on simulation and 61 on DROID. Cosine between the imagined path and the real one, with z from the IDM or from the controller and the real actions.

| World model | Simulation, IDM | Simulation, controller | DROID, IDM | DROID, best controller |
|---|---|---|---|---|
| SSv2 10 h | 0.81 | 0.50 | 0.69 | ≤ 0.09 |
| SSv2 20 h | 0.94 | 0.69 | 0.83 | ≤ 0.09 |

- **Measured:** a model trained on human video follows a simulated robot in part with its real actions. On DROID, no controller drives it, even the one trained on DROID, while z from the IDM does.
- **Measured:** the world model trained on DROID does worse than the SSv2 models: IDM 0.14 on simulation, 0.50 on DROID.
- **Measured:** against a still state after 7 steps, only the 20 h model on simulation wins (error 0.023 against 0.030; IDM 0.005).
- **Measured:** the ratio to the IDM misleads. On the DROID world model it reaches 1.0–1.6 only because z has no effect.

### 15. Does a move transfer from one clip to another?

The command of A runs on B, on the same pairs as experiment 14. Cosine with the path of A, then with the path of an unrelated clip C.

| Command | Simulation, 10 h | Simulation, 20 h | DROID, 10 h | DROID, 20 h |
|---|---|---|---|---|
| z of A | 0.77 / 0.03 | 0.60 / −0.02 | 0.13 | 0.09 |
| Actions of A, through the controller | 0.14 / 0.03 | 0.17 / −0.01 | ≈ 0 | ≈ 0 |

- **Measured:** the actions of A move B only weakly, and the intervals for A and C touch.
- **Inferred:** the controller's path follows the state more than the action. A controller given the state alone would measure this (untested).
- **Inferred:** z works inside the single simulated scene, not across DROID scenes. In one scene, a real transfer and a leak look the same, so this does not prove that z is an action.

### 16. Does planning beat doing nothing?

Offline CEM over 3 actions toward a goal state, on 64 test clips. Δxyz is the L1 error on the total move; a zero plan is not moving at all.

| Data | Controllers | Zero plan, same 64 clips | Zero plan, all clips | Move of another clip |
|---|---|---|---|---|
| DROID | 0.24–0.42 | 0.132 | 0.094 | 0.145 |
| Simulation | 0.09–0.38 | 0.070 | 0.085 | about 0.125 |

- **Measured:** every controller plans worse than the zero plan, and on DROID worse than the move of another clip.
- **Measured:** 64 clips are too few, the zero plan changes from 0.132 to 0.094 on all clips. Planning must be compared on the same clips, not against a fixed value.
- **Inferred:** the CEM pushes toward large actions that lower the imagined cost. This is not tested yet.
- **Next:** the CEM is removed from this branch (last version in `248588a`) and will be rebuilt one step at a time, starting by testing whether the cost ranks the true actions first.
