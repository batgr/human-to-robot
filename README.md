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

## Training

```bash
uv run wandb login
uv run h2r train
uv run h2r train --resume outputs/<name>/<run>
```

## Credits

`src/h2r/models/jepa.py` and `src/h2r/models/module.py` are adapted from
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Copyright (c) 2026 Lucas Maes),
released under the MIT license. The license text is in [`LICENSES/LeWM-MIT.txt`](LICENSES/LeWM-MIT.txt).

Paper:

> Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero.
> _LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels._
> arXiv:2603.19312, 2026.
