# Training pi0.5 on the Munich real-robot data

This is the full recipe, from raw data to a served checkpoint. The LoRA commands were run as written on a 24 GB
RTX 3090; the two full fine-tunes ran with the same steps on a 2x GB200 machine, one GPU per run (section 3).
**Shipped for the Munich session: `pi05_ebim_task2_munich_real/munich_v1/29999`** (full fine-tune, section 5).

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
| `pi05_ebim_task2_munich_real_from_sim` | full fine-tune from the sim checkpoint, EMA 0.99 | 32 | 30,000 | 5,000 |

`pi05_ebim_task2_munich_real_from_sim` starts from the best sim full fine-tune
(`pi05_ebim_task2_fixpos_200_and_v5_from_fft` @ 29999) instead of `pi05_base`. So that every joint lands in the slot
(and state-token position) the sim model learned it in, its data config sets `sim_layout_remap=True`: on the server side
munich16 state/actions are moved into the 20-dim sim layout (L arm 3:10, R arm 10:17, grippers 17:19, base and spine
zero) and actions are mapped back to munich16 on output, so the robot client and its layout check are unchanged.
State grippers (knuckle rad, 0 = open) are negated so they rise with opening like the sim open-fraction; action grippers
(width %) already do. It needs **its own 20-dim norm stats** (section 2) and the sim `params/` on disk:

```bash
P=checkpoints/pi05_ebim_task2_fixpos_200_and_v5_from_fft/pi05_fixpos_200_and_v5_from_fft_full_finetune/29999
hf download Ching-Yeh/ebim-task2 --include "$P/params/*" --include "$P/assets/*" --local-dir .   # ~12 GB, one --include per pattern
```

The real arm poses differ a lot from the sim ones (per-joint medians up to ~2 rad apart), so how much the sim prior
helps is an empirical question: compare it with the `pi05_base` run on the hold-out (section 5).

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

For `pi05_ebim_task2_munich_real_from_sim` compute its own stats (20-dim, not interchangeable with the two above):

```bash
uv run --env-file .env scripts/compute_norm_stats.py --config-name pi05_ebim_task2_munich_real_from_sim
```

Expected, not a bug: left knuckle state (dim 7) has q01 = q99 = 0 and the left gripper action (dim 7) is constant 1.0.
`assets/` is gitignored. Training copies the stats into each checkpoint's `assets/`, and the server and the hold-out
eval read them from there. Recompute and recopy whenever the dataset is converted again.

## 3. Train

Full fine-tunes, as run (one GB200 each, both at the same time; W&B off):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run scripts/train.py pi05_ebim_task2_munich_real \
  --exp-name=munich_v1 --fsdp-devices 1 --no-wandb-enabled --overwrite
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run scripts/train.py pi05_ebim_task2_munich_real_from_sim \
  --exp-name=munich_v1 --fsdp-devices 1 --no-wandb-enabled --overwrite
```

Each ran at about 1.4 it/s at batch 32, so 30,000 steps took about 6 hours (5 h 53 min and 6 h 33 min; the slower
one shared its GPU with another job near the end). A single 189 GB GB200 holds a full fine-tune; how much less would
do was not measured (JAX reserves 75 % of the GPU up front, about 142 GB here). On smaller GPUs shard the model with
`--fsdp-devices N` (N GPUs; the batch size must be divisible by the device count). `PYTHONUNBUFFERED=1` matters
when the output goes through a pipe (`| tee log`): the `Step N: loss=...` lines go to stdout, which Python otherwise
buffers in 8 KB blocks, so they would reach the log only every ~12,000 steps.

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
  last step and multiples of `keep_period` survive. The last step of a 30,000-step run is saved as `29999` (steps
  count from 0), not `30000`.
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
  --checkpoint-dir checkpoints/pi05_ebim_task2_munich_real/munich_v1/29999
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

LoRA stand-in (`pi05_lora_ebim_task2_munich_real/munich_v1`, 321 samples, stride 20):

| Step | arm MAE@10 | arm MAE@25 | arm MAE@50 | gripper MAE@25 | first_step_offset p50 / p95 / max (rad) |
|---|---|---|---|---|---|
| 2500 | 0.0393 | 0.0513 | 0.0665 | 0.0308 | 0.166 / 0.515 / 0.862 |
| 5000 | 0.0344 | 0.0464 | 0.0612 | 0.0243 | 0.129 / 0.374 / 0.649 |
| 7500 | 0.0327 | 0.0443 | 0.0586 | 0.0250 | 0.138 / 0.359 / 0.685 |
| 10000 | 0.0315 | 0.0434 | 0.0583 | 0.0223 | 0.122 / 0.354 / 0.601 |
| **12500** | **0.0298** | **0.0410** | **0.0552** | **0.0205** | 0.113 / **0.309** / 0.484 |
| 14999 | 0.0304 | 0.0417 | 0.0559 | 0.0229 | 0.117 / 0.327 / 0.522 |

No LoRA step met the p95 < 0.3 rad gate (the recorded data itself has p95 0.198 rad between the action and the measured
state on the same frames). Step 12500 has both the lowest arm MAE@25 and the lowest p95, so it was the stand-in shipped before the full fine-tunes.
The gate matters for the client at rest, where it checks the first chunk: on the first frame of each of the 10 hold-out
episodes step 12500's first-step offset was 0.05 to 0.25 rad (max 0.253), inside `max_start_offset_rad`.

Full fine-tunes (`munich_v1`, same hold-out, 321 samples, stride 20). "Missed" counts the 57 ground-truth right-gripper
switches in the sampled chunks that the prediction did not reproduce (the left gripper never switches):

`pi05_ebim_task2_munich_real` (from `pi05_base`):

| Step | arm MAE@10 | arm MAE@25 | arm MAE@50 | gripper MAE@25 | first_step_offset p50 / p95 / max (rad) | R gripper missed |
|---|---|---|---|---|---|---|
| 5000 | 0.0331 | 0.0446 | 0.0595 | 0.0245 | 0.105 / 0.302 / 0.483 | 16 |
| 10000 | 0.0325 | 0.0442 | 0.0589 | 0.0238 | 0.101 / 0.275 / 0.474 | 15 |
| 15000 | 0.0328 | 0.0446 | 0.0595 | 0.0240 | 0.101 / 0.288 / 0.520 | 12 |
| 20000 | 0.0319 | **0.0438** | 0.0586 | 0.0237 | 0.093 / 0.265 / 0.488 | 13 |
| 25000 | 0.0322 | 0.0443 | 0.0590 | 0.0231 | 0.096 / 0.265 / 0.486 | 14 |
| **29999 (shipped)** | 0.0318 | 0.0440 | 0.0589 | 0.0233 | 0.098 / 0.273 / 0.477 | 15 |

`pi05_ebim_task2_munich_real_from_sim` (from the sim checkpoint, sim-slot remap):

| Step | arm MAE@10 | arm MAE@25 | arm MAE@50 | gripper MAE@25 | first_step_offset p50 / p95 / max (rad) | R gripper missed |
|---|---|---|---|---|---|---|
| 5000 | 0.0324 | 0.0436 | 0.0578 | 0.0259 | 0.091 / 0.227 / 0.377 | 16 |
| 10000 | 0.0315 | 0.0429 | 0.0574 | 0.0251 | 0.089 / 0.207 / 0.403 | 17 |
| 15000 | 0.0314 | 0.0430 | 0.0577 | 0.0255 | 0.091 / 0.208 / 0.384 | 20 |
| 20000 | 0.0308 | **0.0422** | 0.0569 | 0.0251 | 0.089 / 0.207 / 0.401 | 18 |
| 25000 | 0.0310 | 0.0426 | 0.0573 | 0.0247 | 0.092 / 0.206 / 0.394 | 18 |
| 29999 | 0.0309 | 0.0426 | 0.0573 | 0.0247 | 0.090 / 0.209 / 0.398 | 18 |

What the numbers say:

- Both full fine-tunes pass the p95 < 0.3 rad gate at every step from 10000 on (the LoRA never did). By the rule above
  the best step is 20000 in both runs.
- The sim-initialised run is ahead on the arms at every step (arm MAE@25 about 4 % lower, first-step offset p95 about
  0.21 against 0.27 rad, max about 0.40 against 0.49 rad) and slightly behind on the gripper (gripper MAE@25 0.0251
  against 0.0237 and 18 against 13 missed switches at step 20000). Training loss was the same for both from step 5000 on
  (0.0012 at the end).
- Neither beats the LoRA stand-in's arm MAE@25 (0.0410 at 12500), but the LoRA fails the offset gate.
- After 10000 steps all numbers move by less than 0.001 rad, so the steps are close to interchangeable on the hold-out.

**Shipped: `pi05_ebim_task2_munich_real/munich_v1/29999`**, the last step of the `pi05_base` run. It was picked
instead of the rule's 20000; on the hold-out the two differ by 0.0002 rad in arm MAE@25 (0.0440 against 0.0438) and
0.008 rad in offset p95 (0.273 against 0.265), and 29999 passes the gate. The other three uploaded full-FT steps
(section 6) are the alternatives for [swapping](#7-serve-and-swap-checkpoints).

## 6. Upload

```bash
examples/ebim_task2/upload_checkpoint.sh pi05_ebim_task2_munich_real munich_v1 29999 --dry-run   # prints the commands
examples/ebim_task2/upload_checkpoint.sh pi05_ebim_task2_munich_real munich_v1 29999
```

Uploads `params/`, `assets/` and `holdout_eval.json` to the HF model repo `hermanprawiro/ebim-task2-real` under
`<config>/<exp>/<step>/` (same layout as on disk; `--repo` selects another repo, `--dry-run` only prints the commands).
`hermanprawiro/ebim-task2-real` is **public**, so an upload there is published at once. A missing repo (another
`--repo`) is created **private**; an existing repo keeps its visibility. It refuses when `params/`, `assets/` (with a
`norm_stats.json`) or `holdout_eval.json` is missing, and never uploads `train_state/`. It uses `hf` if that is on
`PATH`, else `huggingface-cli`, else the repo's `uv run huggingface-cli`. You need `HF_TOKEN` (or `hf auth login`) with
write access first. Export only the token: `.env`'s `HF_HOME=~/...` is not tilde-expanded by `uv run --env-file`, so
`uv run --env-file .env ...` would put the upload cache in a directory literally named `~` and the Xet upload fails
(`MerkleDB Shard error: File I/O error`). For example, with only the token from `.env` exported:

```bash
HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '"') \
  examples/ebim_task2/upload_checkpoint.sh pi05_ebim_task2_munich_real munich_v1 29999
```

The script prints the matching download and serve commands. To fetch a step on another machine (no token needed;
`pip install -U huggingface_hub` provides `hf`; older versions call it `huggingface-cli`):

```bash
hf download hermanprawiro/ebim-task2-real --include "pi05_ebim_task2_munich_real/munich_v1/29999/*" --local-dir checkpoints
```

What is in the public repo (`params/`, `assets/`, `holdout_eval.json` each):

| Config | Steps | Size per step |
|---|---|---|
| `pi05_ebim_task2_munich_real` | **29999 (shipped)**, 20000 | 12.4 GB |
| `pi05_ebim_task2_munich_real_from_sim` | 20000, 29999 | 12.4 GB |
| `pi05_lora_ebim_task2_munich_real` | 12500 (the former LoRA stand-in) | 6.3 GB |

The private repo `hermanprawiro/ebim-task2-real-train` holds what is needed to resume or extend either full fine-tune:
step 29999 of both runs with `train_state/` (about 45 GB per run), every step's `holdout_eval.json` and the training
logs. The other steps' `train_state/` was not uploaded.

## 7. Serve and swap checkpoints

```bash
examples/ebim_task2/serve_munich.sh pi05_ebim_task2_munich_real checkpoints/pi05_ebim_task2_munich_real/munich_v1/29999 [port]
```

This wraps `examples/ebim_task2/compose.policy_server.yml` (build from `scripts/docker/serve_policy.Dockerfile`, repo
bind-mounted at `/app`, checkpoint step dir mounted read-only at `/ckpt/<exp>/<step>`, host networking, one GPU, default
port 8000). It first checks that `<ckpt_dir>/params` and `<ckpt_dir>/assets` exist. The server reports `<exp>/<step>` to
clients as the checkpoint name. Without the wrapper:

```bash
POLICY_CONFIG=pi05_ebim_task2_munich_real CKPT_DIR=$PWD/checkpoints/pi05_ebim_task2_munich_real/munich_v1/29999 \
  CKPT_NAME=munich_v1/29999 PORT=8000 docker compose -f examples/ebim_task2/compose.policy_server.yml up --build
```

**Swapping a checkpoint is server-side only**: stop the container, start it again with another `POLICY_CONFIG`
(full FT or LoRA config) and `CKPT_DIR` (another step). The robot client is unchanged; at connect time it checks the
server metadata (`layout`, dims, 20 Hz, horizon 50 and the prompt) and refuses a mismatching server.

**`POLICY_CONFIG` must be the checkpoint's first directory level.** The two full-FT configs have the same
architecture and the same server metadata, so the weights of either load under the other; without a check, a
`pi05_ebim_task2_munich_real_from_sim` checkpoint served with `pi05_ebim_task2_munich_real` answered with finite
`(50, 16)` chunks, passed the client's check and put the joints in the wrong slots. The server therefore compares the
width of the checkpoint's norm stats with the config's layout (16, or 20 for `_from_sim`) before it loads the model
and stops with `ValueError: state norm stats are 20-dim, but this config expects 16 ...` (or `16-dim ... expects 20`);
both pairings were checked against the real checkpoints. Training does the same check on `assets/`. A LoRA checkpoint
with a full-FT config, or the reverse, fails to load anyway.

The server downloads the PaliGemma tokenizer on first start into `OPENPI_DATA_HOME` (default `~/.cache/openpi`, mounted
as `/openpi_assets`) at `big_vision/paligemma_tokenizer.model`. On a GPU host without internet, pre-fetch it on a
machine with internet and copy the file to the same relative path, or run once there:

```bash
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}))"
```

Starting the server needs no other network access: the compose file sets `UV_NO_SYNC=1` and `PYTHONPATH=/app/src`, so
`uv run` does not re-sync from PyPI in a new container (see `SERVER.md` section 6). The server is ready when the log
shows the warm-up finished (about a minute to load plus ~40 s to compile).

## 8. Troubleshooting

- **OOM at start or mid-run (LoRA on 24 GB):** close other GPU processes (the desktop and a running server also use
  VRAM), set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, or lower `--batch-size 8`. Full FT: raise `--fsdp-devices`, or
  reduce `--batch-size`.
- **Stale norm stats:** if you re-convert the data or change the hold-out seed, recompute and recopy norm stats
  (section 2) before training. A checkpoint trained with other stats than it was evaluated or served with gives
  confident-looking but offset actions; compare `norm_stats_sha256` in `holdout_eval.json` with
  `sha256sum <ckpt>/assets/ebim/task2_munich_real/norm_stats.json`, and the LoRA and full-FT hashes with each other
  (`f7b346a6...` for both). `pi05_ebim_task2_munich_real_from_sim` has its own 20-dim stats (`f036190a...`) by design.
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
