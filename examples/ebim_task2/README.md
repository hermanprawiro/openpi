# Fine-tuning pi0 / pi0.5 on the EBIM task2 dataset

Fine-tunes the pi0 or pi0.5 base model (LoRA) on the Isaac Sim teleoperation datasets recorded
with `ebim/benchmark/task2_isaacsim` (dual-arm Franka FR3 on a mobile base with a vertical
spine lift). 

Task: *"Pick up the thermal pad, place it on the target RAM board, and go back to
starting position."*

All train configs share the same policy transforms and the same 20-dim state/action layout:

| config | base checkpoint | dataset | action space | state input |
|--------|-----------------|---------|--------------|-------------|
| `pi0_ebim_task2_low_mem_finetune` | `pi0_base` | `ebim/task2_thermalpad` | delta arm joints + spine | continuous vector |
| `pi05_ebim_task2_low_mem_finetune` | `pi05_base` | `ebim/task2_thermalpad` | absolute (as recorded) | discrete tokens in the prompt |
| `pi05_ebim_task2_baseline_finetune` | `pi05_base` | `ebim/task2_baseline_v1` | absolute (as recorded) | discrete tokens in the prompt |
| `pi05_ebim_task2_baseline_v2_finetune` | `pi05_base` | `ebim/task2_baseline_v2` | absolute (as recorded) | discrete tokens in the prompt |
| `pi05_ebim_task2_fixpos_v1_finetune` | `pi05_base` | `ebim/task2_fixpos_v1` | absolute (as recorded) | discrete tokens in the prompt |

All are LoRA (`gemma_2b_lora` + `gemma_300m_lora`, EMA off) and fit on a 24 GB GPU.

## Datasets

All raw datasets are LeRobot v3.0 with an identical feature schema. openpi pins an older
lerobot that reads the v2.1 format, so `convert_task2_data_to_lerobot.py` rewrites them.

| raw dir (under `ebim/benchmark/task2_isaacsim/dataset/`) | converted `repo_id` | episodes | frames | notes |
|---|---|---|---|---|
| `task2_thermalpad_v2` | `ebim/task2_thermalpad` | 16 | — | episode 2 is a failed demo and is dropped |
| `task2_baseline_v1` | `ebim/task2_baseline_v1` | 50 | 77,481 | all episodes successful; merged from `task2_thermalpad_v2` and `task2_thermalpad_v3` |
| `task2_baseline_v2` | `ebim/task2_baseline_v2` | 80 | 127,231 | all episodes successful; superset of `task2_baseline_v1` merged with `task2_thermalpad_v4` |
| `task2_fixpos_v1` | `ebim/task2_fixpos_v1` | 22 | 20,869 | all episodes successful; fixed pose robot |

Each dataset gets its own train config rather than repointing an existing one, so previously
trained checkpoints keep resolving to the norm stats they were trained with.

Cameras used: `head` (base view), `wrist_left`, `wrist_right` — letterbox-resized to 224x224.
The static `eval_camera` is excluded.

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

Actions in the raw dataset are absolute post-arbitration targets. pi0 was pre-trained on delta
joint actions, so `LeRobotEbimTask2DataConfig(use_delta_actions=True)` converts arm joints and
spine to deltas relative to the current state and the policy server converts them back to
absolute targets at inference time. pi0.5 was pre-trained on absolute joint targets, so the
pi0.5 config sets `use_delta_actions=False` and trains on the recorded actions directly. Either
way the wire contract is unchanged: the server always returns absolute targets.

Norm stats are per-config (`assets/<config-name>/<repo-id>`), so every config needs its own
`compute_norm_stats.py` run — the pi0 and pi0.5 thermal-pad configs because they normalize
different action distributions, and the baseline config because it is a different dataset.
pi0.5 also uses quantile normalization (openpi does this automatically for non-pi0 models)
since it bins the normalized state into 256 discrete tokens.

The 37-dim raw state is reduced to the 20 dims above (raw indices `[34:37]` base vel,
`[14:28]` joints, `[29:31]` grippers, `[28]` spine); world-frame EE poses and odometry are
dropped since they do not generalize and exceed pi0's 32-dim state limit.

## Usage

### pi0.5 on `task2_baseline_v1`

```bash
# 1. Convert (50 episodes, all successful -> $HF_LEROBOT_HOME/ebim/task2_baseline_v1)
uv run examples/ebim_task2/convert_task2_data_to_lerobot.py \
    --raw-dir /path/to/dataset/task2_baseline_v1 \
    --repo-id ebim/task2_baseline_v1

# 2. Normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_ebim_task2_baseline_finetune

# 3. Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ebim_task2_baseline_finetune \
    --exp-name=pi05_baseline_v1 --overwrite --keep-period 5000

# 4. Serve
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ebim_task2_baseline_finetune \
    --policy.dir=checkpoints/pi05_ebim_task2_baseline_finetune/pi05_baseline_v1/30000
```

### pi0.5 on `task2_fixpos_v1`

```bash
# 1. Convert (22 episodes, all successful -> $HF_LEROBOT_HOME/ebim/task2_fixpos_v1)
uv run examples/ebim_task2/convert_task2_data_to_lerobot.py \
    --raw-dir /path/to/dataset/task2_fixpos_v1 \
    --repo-id ebim/task2_fixpos_v1

# 2. Normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_ebim_task2_fixpos_v1_finetune

# 3. Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ebim_task2_fixpos_v1_finetune \
    --exp-name=pi05_fixpos_v1 --overwrite --keep-period 5000

# 4. Serve
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ebim_task2_fixpos_v1_finetune \
    --policy.dir=checkpoints/pi05_ebim_task2_fixpos_v1_finetune/pi05_fixpos_v1/30000
```

If training OOMs, lower the batch size (`--batch-size 16`).

See [POLICY_SERVER.md](POLICY_SERVER.md) for serving a trained checkpoint via Docker instead.

## Inference client

The policy server expects observations with these keys (see `EbimTask2Inputs`):

- `observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image` —
  RGB uint8 HWC (any resolution; resized server-side)
- `observation/state` — 20-dim float, layout above
- `prompt` — the task instruction string

It returns an action chunk of shape `(50, 20)` (absolute targets, layout above), matching
the recorded action contract of the sim bridge at 30 Hz.
