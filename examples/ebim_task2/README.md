# Fine-tuning pi0.5 / pi0 on the EBIM task2 dataset

Fine-tunes a pi0.5 or pi0 base model on the EBIM task2 Isaac Sim teleoperation dataset:
a dual-arm Franka FR3 on a mobile base with a vertical spine lift, recorded at a fixed robot
pose. Task: *"Pick up the thermal pad and place it on the target RAM board."*

Available train configs (all share the same policy transforms and 20-dim state/action layout,
described in the [technical reference](#technical-reference) below):

| config | base checkpoint | fine-tuning | action space | GPU memory |
|--------|-----------------|-------------|--------------|------------|
| `pi05_ebim_task2_fixpos_200_finetune` **(default)** | `pi05_base` | LoRA | absolute (as recorded) | 24 GB |
| `pi0_ebim_task2_fixpos_200_finetune` | `pi0_base` | LoRA | delta arm joints + spine | 24 GB |
| `pi05_ebim_task2_fixpos_200_full_finetune` | `pi05_base` | full (EMA on) | absolute (as recorded) | 80 GB-class |

The walkthrough below uses the default pi0.5 LoRA config; for the alternatives just swap the
config name (everything else is identical).

## Fine-tuning walkthrough

Starting from a fresh clone of this repo:

### 1. Set up the environment

Install dependencies with [uv](https://docs.astral.sh/uv/) (see the top-level
[README](../../README.md) for details):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Then create your env file and fill in your Hugging Face token:

```bash
cp .env.example .env
```

`.env` also sets the two Hugging Face storage locations: `HF_HOME` — the cache root where the
raw dataset snapshot is downloaded (`~/.cache/huggingface` by default) — and
`HF_LEROBOT_HOME` — the directory where the converted dataset is written
(`~/.cache/huggingface/lerobot` by default). Point them somewhere else if you want the data on
a different disk; `HF_LEROBOT_HOME` in particular must be set to the same value for all
commands below (the training scripts look the dataset up there), which is why every command
uses `uv run --env-file .env`.

> **Note:** training requires Linux with an NVIDIA GPU. On Windows, run everything through
> Docker instead — see [Training with Docker](#training-with-docker).

### 2. Convert the dataset

Downloads the raw LeRobot v3.0 dataset from Hugging Face and rewrites it into the v2.1 format
that openpi's pinned lerobot can read (details in the technical reference):

```bash
uv run --env-file .env examples/ebim_task2/convert_task2_data_to_lerobot.py \
    --raw-repo-id hermanprawiro/task2_fixpos_200 \
    --repo-id ebim/task2_fixpos_200
```

`--raw-repo-id` also accepts the full URL form
(`https://huggingface.co/datasets/<username>/task2_fixpos_200`). If you already have the
raw dataset on disk, use `--raw-dir /path/to/task2_fixpos_200` instead. The converted dataset
is written to `$HF_LEROBOT_HOME/ebim/task2_fixpos_200`, which is where the training scripts
expect it (`--repo-id` must stay `ebim/task2_fixpos_200` to match the train configs).

### 3. Compute normalization statistics

```bash
uv run --env-file .env scripts/compute_norm_stats.py \
    --config-name pi05_ebim_task2_fixpos_200_finetune
```

### 4. Train

Fits on a 24 GB GPU with the config's default batch size of 16:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --env-file .env scripts/train.py \
    pi05_ebim_task2_fixpos_200_finetune \
    --exp-name=pi05_fixpos_200 --overwrite --keep-period 5000
```

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` makes JAX preallocate 90% of the GPU memory instead of
its default 75% — training needs the larger pool, and the remaining 10% leaves headroom for
the CUDA driver and cuDNN workspaces. If training still OOMs, lower the batch size
(`--batch-size 8`).

### 5. Serve the trained policy

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ebim_task2_fixpos_200_finetune \
    --policy.dir=checkpoints/pi05_ebim_task2_fixpos_200_finetune/pi05_fixpos_200/30000
```

See [POLICY_SERVER.md](POLICY_SERVER.md) for serving via Docker instead.

## Training with Docker

JAX training does not run natively on Windows, and this is also the easiest way to get a
reproducible environment on Linux. Build the image once from the repo root:

```bash
docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile
```

Then run each walkthrough step inside the container. The repo is mounted at `/app` and the
Hugging Face cache at `/root/.cache/huggingface`, so the raw-dataset download and the
converted dataset persist on the host across runs. On Windows, run these from Git Bash and
prefix each `docker run` with `MSYS_NO_PATHCONV=1` so MSYS doesn't rewrite the container
paths:

```bash
# 2. Convert
docker run --rm --shm-size=8g \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env-file .env \
  openpi_server /bin/bash -c "uv run examples/ebim_task2/convert_task2_data_to_lerobot.py --raw-repo-id hermanprawiro/task2_fixpos_200 --repo-id ebim/task2_fixpos_200"

# 3. Norm stats
docker run --rm --shm-size=8g \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/openpi:/openpi_assets" \
  --env-file .env -e OPENPI_DATA_HOME=/openpi_assets \
  openpi_server /bin/bash -c "uv run scripts/compute_norm_stats.py --config-name pi05_ebim_task2_fixpos_200_finetune"

# 4. Train
docker run --rm --gpus=all --shm-size=8g \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/openpi:/openpi_assets" \
  --env-file .env -e OPENPI_DATA_HOME=/openpi_assets -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  openpi_server /bin/bash -c "uv run scripts/train.py pi05_ebim_task2_fixpos_200_finetune --exp-name=pi05_fixpos_200 --overwrite --keep-period 5000"
```

Only the train step needs `--gpus=all` (conversion is CPU-only by design, and norm stats runs
fine on CPU). The mounted caches keep the raw-dataset download (`~/.cache/huggingface`) and
the base checkpoint download (`~/.cache/openpi`) across container runs. Add
`--no-wandb-enabled` to the train command if you don't want to log to Weights & Biases.

## Inference client

The policy server expects observations with these keys (see `EbimTask2Inputs`):

- `observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image` —
  RGB uint8 HWC (any resolution; resized server-side)
- `observation/state` — 20-dim float, layout below
- `prompt` — the task instruction string

It returns an action chunk of shape `(50, 20)` (absolute targets, layout below), matching
the recorded action contract of the sim bridge at 30 Hz.

## Technical reference

### Dataset

`task2_fixpos_200` is a raw LeRobot **v3.0** dataset: 200 episodes, 174,719 frames at 30 Hz, 
recorded by `ebim/benchmark/task2_isaacsim` with a fixed robot start pose. openpi pins an 
older lerobot that reads the **v2.1** format, so `convert_task2_data_to_lerobot.py` reads 
the v3.0 parquet/video files directly (pyarrow + pyav) and rewrites them through the pinned 
lerobot as `ebim/task2_fixpos_200`. Episodes labeled `success=false` in 
`task2_extras/episodes_task2.jsonl` are skipped unless `--include-failed` is passed.

Cameras used: `head` (base view), `wrist_left`, `wrist_right` — letterbox-resized to 224x224.
The static `eval_camera` is excluded.

### State/action layout

State and actions share the same 20-dim layout so the delta-action transform is element-wise
consistent:

| dims | content | delta (pi0 only)? |
|------|---------|-------------------|
| 0-2  | base twist vx, vy, wz (m/s, rad/s) | absolute (velocities) |
| 3-9  | left arm joints 1-7 (rad) | delta |
| 10-16 | right arm joints 1-7 (rad) | delta |
| 17 | left gripper open-fraction [0, 1] | absolute |
| 18 | right gripper open-fraction [0, 1] | absolute |
| 19 | spine height (m) | delta |

The 37-dim raw state is reduced to the 20 dims above (raw indices `[34:37]` base vel,
`[14:28]` joints, `[29:31]` grippers, `[28]` spine); world-frame EE poses and odometry are
dropped since they do not generalize and exceed pi0's 32-dim state limit.

### Delta vs. absolute actions

Actions in the raw dataset are absolute post-arbitration targets. pi0 was pre-trained on delta
joint actions, so `LeRobotEbimTask2DataConfig(use_delta_actions=True)` converts arm joints and
spine to deltas relative to the current state and the policy server converts them back to
absolute targets at inference time. pi0.5 was pre-trained on absolute joint targets, so the
pi0.5 configs set `use_delta_actions=False` and train on the recorded actions directly. Either
way the wire contract is unchanged: the server always returns absolute targets.

### Normalization statistics

Norm stats are per-config (`assets/<config-name>/<repo-id>`), so each config needs its own
`compute_norm_stats.py` run — the pi0 and pi0.5 configs normalize different action
distributions (delta vs. absolute). pi0.5 also uses quantile normalization (openpi does this
automatically for non-pi0 models) since it bins the normalized state into 256 discrete tokens.
