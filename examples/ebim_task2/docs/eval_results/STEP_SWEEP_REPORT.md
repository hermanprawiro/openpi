# EBiM Task 2: checkpoint step sweep

_Final, 2026-10-01. All four configs are scored at 5k–30k in all three client modes (72 runs, 720 episodes)._

## Setup

- **Eval:** 10 episodes per run, 90 sim-seconds each, Isaac Sim room scene with `--randomize-objects --robot-x 2.1 --robot-y 3.05`. The target board slot (A–D) is re-sampled on every reset; B is the home slot.
- **Success:** evaluator IoU > 0 and orientation correct. Per-slot successes are in parentheses.
- **Client modes:** sync = `--no-async-inference`; async replace = `--chunk-merge replace`; async blend-old = `--chunk-merge blend_favor_old`. Everything else comes from `eval_config.yaml`.
- **Serving:** local `openpi_server` container on the same RTX 3090 as the sim, capped at 35% GPU memory. Sync inference averaged about 320 ms. The 30k rows and `200_and_v5_from_fft` sync 20k were run earlier against the remote server.
- **Noise:** with 10 episodes, a 95% interval on 2/10 is roughly 3–56%. Differences of one or two successes between cells are not meaningful. Compare totals out of 30, and read them together with the slot columns (see "Slot mix confounds the totals" below).
- **Slot columns:** successes / episodes on the home slot B and on the other slots, summed over the scored modes of that row.

## Results

### `pi05_lora_ebim_task2_fixpos_v5_from_fft`

| Step | Sync | Async replace | Async blend-old | Total | Slot B | Slots A/C/D | Sync off-board |
|---|---|---|---|---|---|---|---|
| 5k | 0/10 | 1/10 (C1) | 0/10 | **1/30** | 0/8 | 1/22 | 8/10 |
| 10k | 0/10 | 1/10 (C1) | 0/10 | **1/30** | 0/11 | 1/19 | 3/10 |
| 15k | 0/10 | 0/10 | 1/10 (C1) | **1/30** | 0/9 | 1/21 | 5/10 |
| 20k | 3/10 (B2, C1) | 0/10 | 0/10 | **3/30** | 2/8 | 1/22 | 2/10 |
| 25k | 2/10 (A1, D1) | 0/10 | 2/10 (C1, D1) | **4/30** | 0/9 | 4/21 | 2/10 |
| 30k | 1/10 (B1) | 1/10 (D1) | 1/10 (D1) | **3/30** | 1/7 | 2/23 | 3/10 |

### `pi05_lora_ebim_task2_fixpos_200_and_v5_from_fft`

| Step | Sync | Async replace | Async blend-old | Total | Slot B | Slots A/C/D | Sync off-board |
|---|---|---|---|---|---|---|---|
| 5k | 2/10 (B2) | 2/10 (B2) | 2/10 (B2) | **6/30** | 6/12 | 0/18 | 0/10 |
| 10k | 4/10 (B2, C1, D1) | 3/10 (B2, C1) | 3/10 (B3) | **10/30** | 7/7 | 3/23 | 0/10 |
| 15k | 2/10 (B2) | 2/10 (B2) | 3/10 (B1, C2) | **7/30** | 5/7 | 2/23 | 1/10 |
| 20k | 2/10 (B1, D1) | 1/10 (B1) | 1/10 (B1) | **4/30** | 3/6 | 1/24 | 0/10 |
| 25k | 1/10 (A1) | 2/10 (B2) | 1/10 (D1) | **4/30** | 2/6 | 2/24 | 0/10 |
| 30k | 5/10 (A1, B3, C1) | 3/10 (B2, D1) | 3/10 (B3) | **11/30** | 8/11 | 3/19 | 0/10 |

### `pi05_lora_ebim_task2_fixpos_200_and_v5`

| Step | Sync | Async replace | Async blend-old | Total | Slot B | Slots A/C/D | Sync off-board |
|---|---|---|---|---|---|---|---|
| 5k | 0/10 | 0/10 | 0/10 | **0/30** | 0/6 | 0/24 | 10/10 |
| 10k | 0/10 | 0/10 | 0/10 | **0/30** | 0/9 | 0/21 | 10/10 |
| 15k | 0/10 | 0/10 | 0/10 | **0/30** | 0/5 | 0/25 | 10/10 |
| 20k | 0/10 | 0/10 | 0/10 | **0/30** | 0/5 | 0/25 | 9/10 |
| 25k | 0/10 | 0/10 | 0/10 | **0/30** | 0/5 | 0/25 | 10/10 |
| 30k | 0/10 | 0/10 | 0/10 | **0/30** | 0/7 | 0/23 | 6/10 |

### `pi05_ebim_task2_fixpos_200_full_finetune`

| Step | Sync | Async replace | Async blend-old | Total | Slot B | Slots A/C/D | Sync off-board |
|---|---|---|---|---|---|---|---|
| 5k | 1/10 (B1) | 0/10 | 1/10 (B1) | **2/30** | 2/7 | 0/23 | 7/10 |
| 10k | 1/10 (B1) | 1/10 (C1) | 0/10 | **2/30** | 1/5 | 1/25 | 2/10 |
| 15k | 3/10 (B3) | 1/10 (B1) | 0/10 | **4/30** | 4/9 | 0/21 | 1/10 |
| 20k | 2/10 (B2) | 0/10 | 1/10 (B1) | **3/30** | 3/5 | 0/25 | 1/10 |
| 25k | 1/10 (B1) | 2/10 (B2) | 4/10 (B4) | **7/30** | 7/9 | 0/21 | 0/10 |
| 30k | 3/10 (B3) | 1/10 (B1) | 4/10 (B4) | **8/30** | 8/11 | 0/19 | 0/10 |

## Findings

### Which checkpoint to use

- **`200_and_v5_from_fft` is the strongest config at every step.** Its totals range from 4/30 to 11/30. Its sync runs almost never drop the pad off-board.
- **Its best checkpoints are 10k and 30k, and 10k is at least as good as 30k.** The raw totals (10/30 vs 11/30) favour 30k, but 30k drew slot B in 11 of its 30 episodes against 7 for 10k. Split by slot, 10k went 7/7 on B and 3/23 elsewhere; 30k went 8/11 on B and 3/19 elsewhere. 20k and 25k are worse on both B (3/6, 2/6) and non-B (1/24, 2/24). An older 120k checkpoint of the same config scored 8/30 (B 7/13, non-B 1/17), so training longer does not help.
- **`fixpos_v5_from_fft` stays at 1–4/30 at all steps.** Sync runs at 5k–15k leave the pad off-board in 3–8 of 10 episodes, so pickup is the main failure. It improves slightly after 20k.
- **Plain `200_and_v5` (LoRA from `pi05_base`) scores 0 at every step evaluated.** Its sync runs never lift the pad at 5k–15k (10/10 off-board). Initializing from the full-finetune checkpoint is what makes the combined dataset work.
- **The full finetune on `fixpos_200` alone (the init for both `_from_fft` configs) almost never succeeds off slot B.** Over 18 runs it went 25/46 on B and 1/134 elsewhere; the one exception is a single slot-C episode. Its B rate (54%) is close to `200_and_v5_from_fft`'s 61%, so the 20 v5 demos did not cost any B performance, and nearly all of `200_and_v5_from_fft`'s non-B successes come from them. Pickup becomes reliable from 15k on (0–1/10 off-board in sync), and the totals rise with steps (2/30 at 5k, 8/30 at 30k), all from B.
- **Async modes do not beat sync.** Summed over the three configs that ever succeed and all six steps, sync scores 33/180, async replace 21/180, and async blend-old 27/180. Blend-old is close to sync; replace is consistently the weakest. The 30k crossfade and offset-decay runs scored 0–1/10 each.

### Where the policy fails

Pooled over every scored episode of each config (all steps and modes):

| Outcome | `200_and_v5_from_fft` (220 ep.) | `fixpos_v5_from_fft` (200 ep.) | `200_and_v5` (190 ep.) | full finetune (180 ep.) |
|---|---|---|---|---|
| Success | 23% | 7% | 1% | 14% |
| Pad off-board (dropped or never lifted) | 15% | 42% | 93% | 28% |
| Pad on the board, IoU 0 with target | 38% | 25% | 1% | 42% |
| Wrong orientation | 24% | 27% | 6% | 16% |
| Success rate, slot B | **39/64 (61%)** | 3/59 (5%) | 0/38 | 25/46 (54%) |
| Success rate, slots A/C/D | **12/156 (8%)** | 11/137 (8%) | 1/152 | **1/134 (1%)** |
| Median IoU of successes | 0.40 | 0.21 | 0.04 | 0.44 |

- **The best config drives the pad to slot B no matter which slot is the target.** Taking the eval-camera pixel position where the pad came to rest and finding the nearest slot: when the target was A, 25 of 35 on-board pads ended at B; when the target was C, 35 of 49 ended at B. For target B, 50 of 53 ended at B. The full finetune is even more extreme: when the target was A, C or D, 84 of its 95 on-board pads ended at B. Misses land about 9–13 cm from the target centre, roughly one slot pitch.
- **The cause is the training data mix.** `task2_fixpos_200` has 200 demos with no slot randomization (home slot B). `task2_fixpos_v5` has 20 randomized demos, whose reset metadata lists target slots A 5, B 3, C 7, D 8. So non-B placements are about 8% of the combined dataset. The prompt is the same for every episode ("Pick up the thermal pad and place it on the target RAM board."), so the target can only come from vision, and the B prior wins. The full-finetune results confirm it: trained on `fixpos_200` alone, the model succeeds off B once in 134 episodes.
- **`fixpos_v5_from_fft` has the opposite problem.** It learned no home-slot bias, but with only 20 demos it neither picks up reliably (42% off-board) nor places precisely (success IoU 0.21, yaw error about 16° vs 5° for the best config). Its placements drift one slot toward D.
- **Even successful placements are loose.** Median IoU of successes is 0.40 with median spill 0.38, so about a third of the pad lies outside the target.

### Slot mix confounds the totals

Each run samples slots at random, so the share of B episodes per cell ranges from 17% to 43%. For the best config the B success rate (~61%) is about eight times the non-B rate (~8%). One extra B episode is therefore worth about half a success, which is the same size as most of the step-to-step differences. Rank checkpoints on per-slot rates, not raw totals.

## No-randomization check (30k, 2026-10-03)

Same eval, but the scene is launched without `--randomize-objects`: the target board stays at home slot B and nothing is jittered (episode manifests record `randomized: false`). One run per model and mode, 10 episodes each, at step 30k. The last column is each model's slot-B result from the randomized sweep at 30k (all three modes pooled), for comparison.

| Model | Sync | Async replace | Off-board (sync / async) | Mean IoU (sync / async) | Randomized sweep, slot B at 30k |
|---|---|---|---|---|---|
| `200_and_v5_from_fft` | **8/10** | **9/10** | 2 / 1 | 0.42 / 0.53 | 8/11 |
| full finetune (`fixpos_200`) | 7/10 | 4/10 | 0 / 1 | 0.42 / 0.30 | 8/11 |
| `fixpos_v5_from_fft` | 3/10 | 1/10 | 2 / 3 | 0.03 / 0.06 | 1/7 |
| `200_and_v5` | 1/10 | 0/10 | 9 / 9 | 0.03 / 0.00 | 0/7 |

- **With the board fixed at home, the two models initialised from the full finetune succeed 70–90% of the time.** This matches their slot-B rate in the randomized sweep (8/11 each), so the small XY jitter of randomization is not what hurts them. Their failures under randomization come from the target being moved to another slot.
- **`200_and_v5_from_fft` is the best model in both settings,** and async replace did not hurt it here (9/10 vs 8/10 sync). The full finetune dropped from 7/10 to 4/10 under async replace, consistent with async replace being the weakest mode in the sweep.
- **`fixpos_v5_from_fft` fails even at home (1–3/10, mean IoU ≤ 0.06).** Its weakness is placement quality, not just slot choice. With only 20 demos it does not learn a precise placement.
- **Plain `200_and_v5` still cannot pick up the pad** (9/10 off-board in both modes), so its zeros are a pickup failure, independent of randomization.

## Improvement suggestions

Ordered by expected payoff per unit of effort.

1. **Rebalance the training mix toward non-B placements.** Collect more randomized-slot demos, about 30–50 per slot, and/or oversample the v5 episodes (or downsample `fixpos_200`) so that non-B placements are 50–75% of training samples. The home-slot bias is the largest single failure, and the roughly 17 non-B demos we have cannot counter 200 B demos. `200_and_v5_from_fft` shows that the full-finetune initialization already provides reliable pickup, so new demos can focus on placement.
2. **Check that the target board is visually identifiable, and consider putting the slot in the prompt.** With a fixed prompt, the policy must work out which board is the target from the head and wrist images. If nothing visible marks the target board (only the sim's semantic label does), the task is ambiguous from the policy's point of view, and no amount of data will fix non-B placement. Either add a visual cue or condition on the slot, e.g. "…place it on the RAM board in slot C", in both training and eval. `prompt_from_task=True` is already set, so this only needs a per-episode task string in the converted dataset.
3. **Run a stratified, seeded eval protocol.** Fix the slot per episode (e.g. A, B, C, D, A, B, …, 12 or 20 episodes) with seeded offsets, so every checkpoint sees the same conditions. Report B and non-B success separately. This removes the slot-mix confound and makes 10–20-episode cells comparable.
4. **Evaluate sync only when selecting checkpoints.** Across 180 episodes per mode (three configs × six steps), neither async mode beat sync. Dropping them makes a sweep three times cheaper (about 70 min per cell instead of 3.5 h). Keep async for a final latency check on the chosen checkpoint.
5. **Use 10k or 30k of `200_and_v5_from_fft` as the current best, and stop training runs at about 10–15k.** Returns do not grow with steps (20k–25k and 120k are worse). At batch 16, 10k steps is already about 0.8 passes over the 192k frames, and the 20 v5 demos are only 9% of those frames. Save GPU time for data and conditioning experiments rather than longer schedules.
6. **Target placement precision after slot selection works.** Even successful B placements have median IoU 0.4 and spill 0.38. Once non-B selection is fixed, look at the last ~2 s of the demos, e.g. a slower press/release phase or an extra wrist-camera view if placement relies on it, and compare the demo IoU distribution to the policy's.

## Infrastructure notes

- The Isaac container loses GPU access whenever systemd reloads (snapd auto-refresh triggers it). The renderer then floods `vkGetMemoryFdKHR failed`, and the evaluator fails every call with "Sync failed ... semantic_segmentation last=NEVER". Three 15k runs of `fixpos_v5_from_fft` were lost this way and rerun.
- The batch driver restarts the container and scene before every run and retries a run once if any evaluator call fails. A permanent fix needs root: pass `/dev/nvidia*` with `--device` when creating the container, or disable snapd auto-refresh.
- A host reboot on 2026-09-25 interrupted the sweep; the partial run was kept as `pi05_lora_fixpos_200_and_v5_20k_async_replace_partial_reboot`.
- **The system disk fills up.** Each run writes about 1.2 GB of raw recordings (mostly 97 MB of MP4 per episode) to `ebim/benchmark/task2_isaacsim/services/eval_recording/eval_raw_out`, which already holds 72 GB, and `/` had 4.7 GB free on 2026-09-29. The batch driver now stops when less than 3 GB is free. On 2026-09-29 it stopped the sweep with 2.1 GB left; the sweep resumed on 2026-09-30 after space was freed.
- Unscorable or interrupted runs are kept with suffixes `_evaluator_failed`, `_partial_hup`, `_partial_reboot`, or `_stale` and are excluded here.
