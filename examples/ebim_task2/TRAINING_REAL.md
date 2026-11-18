# Training pi0.5 on the Munich real-robot data

This is the full recipe, from raw data to a served checkpoint. Every command below was run as written
on the 24 GB RTX 3090 stand-in (LoRA). The real full fine-tune uses the same steps with the full-FT config.

Fixed contract (do not change between training and serving):

- Layout `munich16`: 16-dim state `[L 7 measured rad, L knuckle rad, R 7 measured rad, R knuckle rad]`,
  16-dim action `[L 7 target rad, L width %, R 7 target rad, R width %]`. Raw dataset values, no unit conversion.
- Absolute actions, action horizon 50, 20 Hz.
- Prompt (exact, no trailing period): `Pick up the thermal pad and place it on the target RAM board`.
- Images: head, left wrist, right wrist at 224x224 RGB (letterboxed). The head video is 1280x720 in 136 of the
  149 valid episodes and 336x188 in 13; the converter resizes every head frame to 336x188 first, then letterboxes
  to 224x224, which is exactly what the robot client does at run time.

Configs (both in `src/openpi/training/config.py`, same data, same prompt, same metadata):

| Config | Mode | Batch | Steps | Save every |
|---|---|---|---|---|
| `pi05_ebim_task2_munich_real` | full fine-tune, EMA 0.99 | 32 | 30,000 | 5,000 |
| `pi05_lora_ebim_task2_munich_real` | LoRA, no EMA | 16 | 15,000 | 2,500 |

## 0. Prerequisites

- `.env` in the repo root with `HF_TOKEN`, `HF_HOME` and `HF_LEROBOT_HOME` (converted datasets, norm stats and
  training all must use the same `HF_LEROBOT_HOME`). All commands use `uv run --env-file .env`.
- Disk: converted dataset about 7.1 GB. Checkpoints: LoRA about 9 GB per step (6 GB `params` + 3 GB `train_state`);
  full FT about 12 GB `params` plus roughly 3x that for `train_state` per step, so budget 50 GB per step and
  about 300 GB for the six saves. `pi05_base` (12.5 GiB) is downloaded once to `~/.cache/openpi` on the first run.
  `checkpoints/` may be a symlink to a bigger disk.
- GPU memory: LoRA fits a 24 GB card at batch 16 (about 23.4 GB used, 5.2 s/step, so 15k steps is about 22 h; run with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`). Full fine-tune needs an 80 GB-class GPU or several GPUs: add
  `--fsdp-devices N` (the model is sharded over N devices, the rest of the devices do data parallel; the batch size
  must be divisible by the device count).
- Raw Munich dataset (LeRobot-style: `meta/`, `data/chunk-000/`, `videos/chunk-000/`) available locally.

## 1. Convert

```bash
uv run --env-file .env examples/ebim_task2/convert_task2_real_to_lerobot.py \
  --raw-dir <munich raw dir> \
  --repo-id ebim/task2_munich_real \
  --holdout-repo-id ebim/task2_munich_real_holdout \
  --num-holdout 10 --seed 0
```

Takes about 20 minutes. Expected: 149 valid / 89 invalid episodes, so **139 train episodes (80,123 frames)** in
`ebim/task2_munich_real` and **10 hold-out episodes (6,321 frames)** in `ebim/task2_munich_real_holdout`.
The same `--seed` always selects the same hold-out episodes; keep it fixed so hold-out numbers stay comparable.
Re-running deletes and recreates both repos. Check the report:

- `$HF_LEROBOT_HOME/ebim/task2_munich_real/conversion/conversion_report.json` (per-episode frame counts, warnings,
  `head_native_wh_counts`)
- `$HF_LEROBOT_HOME/ebim/task2_munich_real/conversion/start_pose_stats.json` (train-only start poses)

`--max-episodes N` makes a quick smoke-test dataset; do not train on it.

## 2. Norm stats

Compute once and give both configs the same file. `compute_norm_stats.py` batches by the config's `batch_size`, so
running it separately per config gives slightly different numbers; copying makes the two configs identical.

```bash
uv run --env-file .env scripts/compute_norm_stats.py --config-name pi05_lora_ebim_task2_munich_real

mkdir -p assets/pi05_ebim_task2_munich_real/ebim/task2_munich_real
cp assets/pi05_lora_ebim_task2_munich_real/ebim/task2_munich_real/norm_stats.json \
   assets/pi05_ebim_task2_munich_real/ebim/task2_munich_real/norm_stats.json
sha256sum assets/pi05*_munich_real/ebim/task2_munich_real/norm_stats.json   # the two hashes must match
```

Expected, not a bug: left knuckle state (dim 7) has q01 = q99 = 0 and the left gripper action (dim 7) is constant 1.0.
`assets/` is gitignored. Training copies the stats into each checkpoint's `assets/`, and the server and the hold-out
eval read them from there. Recompute and recopy whenever the dataset is converted again.

## 3. Train

Full fine-tune (the real run):

```bash
uv run --env-file .env scripts/train.py pi05_ebim_task2_munich_real --exp-name=munich_v1 --fsdp-devices 8 --overwrite
```

LoRA (24 GB stand-in, also usable as a quick baseline):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run --env-file .env scripts/train.py pi05_lora_ebim_task2_munich_real --exp-name=munich_v1 --overwrite
```

Notes:

- Any `TrainConfig` field can be overridden on the command line, for example `--num-train-steps 20000`,
  `--batch-size 64`, `--save-interval 2500`, `--no-wandb-enabled`. Use `--fsdp-devices 1` (default) on a single GPU.
- `--overwrite` wipes `checkpoints/<config>/<exp-name>/`; to continue an interrupted run use `--resume` instead
  (the two are mutually exclusive).
- Checkpoints land in `checkpoints/<config>/<exp>/<step>/` with `params/`, `assets/` and `train_state/`. Only the
  last step and multiples of `keep_period` survive.
- Weights & Biases logging is on by default (project `openpi`).
- Run it under `nohup`/`tmux`; a full 30k-step run takes long enough to need it.

## 4. What to check in the first log lines

1. `Initialized data loader` lists, for batch B: images `(B, 224, 224, 3)` for `base_0_rgb`, `left_wrist_0_rgb`,
   `right_wrist_0_rgb`, `state (B, 32)`, `tokenized_prompt (B, 200)`, and actions `(B, 50, 32)`. The 16 real dims are
   zero-padded to the model's 32; seeing 16 here or an action horizon other than 50 means a wrong config.
2. `Restoring checkpoint from .../pi05_base/params` and `Finished restoring checkpoint`: the base weights loaded.
3. `Step 0: ...`, then one `Step N: loss=... grad_norm=...` line every 100 steps: loss finite (not `nan`/`inf`) and
   clearly falling within the first 500 steps; `grad_norm` finite.
4. The tqdm line gives s/step. On the 3090 LoRA run it is 5.2 s/it at batch 16 and the GPU is the bottleneck. If it
   is several times slower, suspect a CPU fallback or a starved data loader; check `nvidia-smi` utilisation.
5. A `Found 0 checkpoint steps in ...` line on a fresh run is normal.

## 5. Hold-out eval and choosing a step

For each saved step you want to compare (the GPU must be free; it loads the model):

```bash
uv run examples/ebim_task2/eval_real_holdout.py \
  --config-name pi05_ebim_task2_munich_real \
  --checkpoint-dir checkpoints/pi05_ebim_task2_munich_real/munich_v1/30000
```

Optional flags: `--holdout-repo-id` (default `ebim/task2_munich_real_holdout`), `--stride` (default 20 frames between
evaluated samples), `--out` (default `<checkpoint-dir>/holdout_eval.json`).

`holdout_eval.json` holds, per horizon `"10"`, `"25"`, `"50"`: `mae_per_dim`, `arm_mae` and `gripper_mae`; per-arm
`gripper_timing`; `first_step_offset` p50/p95/max (distance between the first predicted action and the current
state); and the `norm_stats_sha256` the model was evaluated with.

**Choose the step with the lowest `arm_mae` at horizon 25, among the steps whose `first_step_offset` p95 is below
`max_start_offset_rad` = 0.3** (the real client refuses to start a chunk further from the robot than that).
Check `gripper_timing` and `gripper_mae` as tie-breakers. The hold-out is open-loop teacher forcing, so it ranks
steps but does not predict closed-loop success; always do a supervised dry run on the robot.

## 6. Upload

```bash
examples/ebim_task2/upload_checkpoint.sh pi05_ebim_task2_munich_real munich_v1 30000
```

Uploads `params/`, `assets/` and `holdout_eval.json` to the HF repo `Ching-Yeh/ebim-task2` under
`<config>/<exp>/<step>/` (same layout as on disk; `--repo` selects another repo, `--dry-run` only prints the commands).
It refuses when `params/`, `assets/` (with a `norm_stats.json`) or `holdout_eval.json` is missing, and never uploads
`train_state/`. You need `huggingface-cli login` (or `HF_TOKEN`) with write access first. It prints the matching
download and serve commands. To fetch a step on another machine:

```bash
huggingface-cli download Ching-Yeh/ebim-task2 --include "pi05_ebim_task2_munich_real/munich_v1/30000/*" --local-dir checkpoints
```

## 7. Serve and swap checkpoints

```bash
examples/ebim_task2/serve_munich.sh pi05_ebim_task2_munich_real checkpoints/pi05_ebim_task2_munich_real/munich_v1/30000 [port]
```

This wraps `examples/ebim_task2/compose.policy_server.yml` (build from `scripts/docker/serve_policy.Dockerfile`, repo
bind-mounted at `/app`, checkpoint step dir mounted read-only at `/ckpt`, host networking, one GPU, default port
8000). It first checks that `<ckpt_dir>/params` and `<ckpt_dir>/assets` exist. Without the wrapper:

```bash
POLICY_CONFIG=pi05_ebim_task2_munich_real CKPT_DIR=$PWD/checkpoints/pi05_ebim_task2_munich_real/munich_v1/30000 \
  PORT=8000 docker compose -f examples/ebim_task2/compose.policy_server.yml up --build
```

**Swapping a checkpoint is server-side only**: stop the container, start it again with another `POLICY_CONFIG`
(full FT or LoRA config) and `CKPT_DIR` (another step). The robot client is unchanged; at connect time it checks the
server metadata (`layout`, dims, 20 Hz, horizon 50 and the prompt) and refuses a mismatching server.

The server downloads the PaliGemma tokenizer on first start into `OPENPI_DATA_HOME` (default `~/.cache/openpi`, mounted
as `/openpi_assets`) at `big_vision/paligemma_tokenizer.model`. On a GPU host without internet, pre-fetch it on a
machine with internet and copy the file to the same relative path, or run once there:

```bash
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}))"
```

The server is ready when the log shows the warm-up finished (about a minute to load plus ~40 s to compile).

## 8. Troubleshooting

- **OOM at start or mid-run (LoRA on 24 GB):** close other GPU processes (the desktop and a running server also use
  VRAM), set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, or lower `--batch-size 8`. Full FT: raise `--fsdp-devices`, or
  reduce `--batch-size`.
- **Stale norm stats:** if you re-convert the data or change the hold-out seed, recompute and recopy norm stats
  (section 2) before training. A checkpoint trained with other stats than it was evaluated or served with gives
  confident-looking but offset actions; compare `norm_stats_sha256` in `holdout_eval.json` with
  `sha256sum <ckpt>/assets/ebim/task2_munich_real/norm_stats.json`, and the LoRA and full-FT hashes with each other.
  If training fails with a missing `norm_stats.json`, the copy for that config name was skipped.
- **Prompt mismatch refusal:** the client exits at connect with a `LayoutMismatchError` such as `prompt: server has
  ..., client uses ...` or `server is not a munich16 policy`. The server config's `policy_metadata` is not the
  Munich one (wrong `POLICY_CONFIG`, e.g. a sim config) or the prompt text differs from
  `Pick up the thermal pad and place it on the target RAM board`. Serve a `*_munich_real` config; do not edit the
  prompt in only one place (`config.py` `_MUNICH_REAL_METADATA` and `robot_client/layout.py` are guarded by a test).
- **Server cannot download the tokenizer:** see the pre-fetch in section 7.
- **`docker compose` says `CKPT_DIR`/`POLICY_CONFIG` is missing:** both are required; use `serve_munich.sh`.
- **Container loses the GPU after a few hours** (`NVML: Unknown Error`) when snapd or systemd reloads the daemon:
  restart the container.
