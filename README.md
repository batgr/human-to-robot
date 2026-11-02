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

# robosuite viewer (configs/env/config.yaml)
uv run python src/h2r/envs/robosuite_env.py

# tests
uv run pytest
```
