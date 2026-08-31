# Chunk merge strategies for async inference

How the eval client reconciles a freshly inferred action chunk with the
actions it has not executed yet, which strategies are implemented
(`chunk_merge` in `eval_config.yaml`), which were surveyed and left out,
and how to compare them. Code: `eval_client/chunking.py`.

## 1. The problem

The policy emits 50-row chunks (1.67 s at 30 Hz). The client executes one
row per control step and, when the queue drops to
`chunk_size_threshold × execute_horizon` rows (0.4 × 50 = 20 by default),
snapshots an observation and asks the server for a new chunk. Inference
takes `L` control steps; the arriving chunk's first `L` rows are stale and
skipped (`chunk_latency_steps_*` in the episode record), so the usable
chunk overlaps the remaining queue by about `20 − L` rows.

The two plans disagree, for three different reasons:

1. **Sampling noise** — flow-matching draws a new sample each time, so even
   from an identical observation the chunk differs slightly.
2. **Genuine reaction** — the new observation shows something the old plan
   did not (object moved, gripper missed); the new plan *should* win.
3. **Mode switch** — a different but equally valid way to do the task
   (approach from the other side). Averaging two modes can produce a plan
   that is neither (the classic argument against temporal ensembling for
   multimodal policies).

Whatever the reason, the executed action stream jumps at the hand-off.
Because the arm dims are absolute joint targets, a jump is a step input
to the joint controllers: visible jerk, and in the worst case oscillation
when consecutive chunks alternate between modes.

**Per-dimension rule.** Only arm joints (dims 3–16) and spine height (19)
are ever interpolated (`BLEND_ACTION_DIMS` in `constants.py`). Base twist
(0–2) is quantized to pedal tokens with a dead-band and the grippers
(17–18) are effectively binary in the dataset, so every strategy takes the
fresh chunk's value for those dims — averaging them yields off-grid twists
and half-open grippers.

**Measuring it.** Every episode record carries
`step_jump_mean/max` (max |Δ| over arm joints between consecutive
published actions, rad) and `switch_jump_mean/max` (the same quantity at
the first action popped after a merge). A strategy that works makes
`switch_jump_*` approach `step_jump_*` without hurting success.

## 2. Implemented strategies

Notation: `old[t]` is the remaining queue (t = 0 is the next step to
publish), `new[t]` the usable fresh chunk aligned to the same steps,
`B = chunk_blend_steps` (default 10), `overlap = min(len(old), len(new))`.

| `chunk_merge` | Rule | Continuous at arrival? | Reactivity | Dry queue |
|---|---|---|---|---|
| `replace` | queue := new | no (full jump) | immediate | n/a |
| `blend_uniform` / `blend_favor_new` / `blend_favor_old` | `w_old·old[t] + w_new·new[t]` over the overlap, then the longer tail | no (jump at arrival *and* where the overlap ends) | partial inside the overlap | degrades to `replace` |
| `crossfade` | `(1−α_t)·old[t] + α_t·new[t]`, `α_t = (t+1)/(Bʹ+1)`, `Bʹ = min(B, overlap)`, then `new` | yes, at both ends of the window | full after `Bʹ` rows | degrades to `replace` |
| `offset_decay` | `new[t] + δ·max(0, 1 − t/D)`, `δ = anchor − new[0]`, `D = min(B, len(new))`, anchor = `old[0]` or the held last action | yes | immediate (the *shape* of `new` is kept) | anchors on the last published action |

- **`replace`** is the "naive async" baseline of the literature. Use it as the
  reference when measuring anything else.
- **`blend_*`** follows LeRobot's `aggregate_fn` (their `weighted_average`).
  Cheap, but the fixed weights only shrink the jump, and averaging two
  plans for up to 20 rows blends modes.
- **`crossfade`** is LiPo's boundary rule and the weight shape LeRobot's
  docs recommend: continuous hand-off, then the fresh plan. The blend
  window still mixes plans; keep `B` short (≈ 5–10 rows) if mode averaging
  is suspected.
- **`offset_decay`** (recommended starting point) adopts the fresh plan
  wholesale and only removes the positional discontinuity, so velocity
  profile and mode of the fresh chunk survive. It is also the only
  implemented strategy that stays continuous when the queue ran dry. A
  lite version of the C¹/C² stitching in VLA-RAIL and Hy-Embodied-0.5.

Selecting: `chunk_merge: offset_decay` + `chunk_blend_steps: 10` in
`eval_config.yaml`, or `EVAL_ARGS="--chunk-merge offset_decay --chunk-blend-steps 10"`.

## 3. Surveyed, not implemented (client-side, training-free)

- **Velocity-matched spline bridge.** Replace the first `c` rows after the
  hand-off with a curve that matches position *and* velocity at both ends:
  cubic Hermite / tangent-aligned cubic Bézier (Hy-Embodied-0.5:
  `P1 = P0 + λ·d̂_hist`, `P2 = P3 − λ·d̂_fut`, position + SLERP + linear
  gripper), or a dual-quintic with C² continuity (VLA-RAIL). Strictly
  smoother than `offset_decay` when old and new plans move in different
  directions; costs two more knobs (window, tangent scale) and can
  overshoot with large velocities. Natural next step if `switch_jump_*`
  is fixed but boundary jerk is still visible.
- **Gated switching (hysteresis).** If `max|old − new|` over the overlap is
  below `ε`, keep the old plan and only append the fresh rows beyond the
  overlap; otherwise merge. Suppresses re-sampling flicker (reason 1
  above) at the cost of ignoring small genuine corrections. A single-sample
  version of BID's *backward coherence*.
- **Commit-then-switch (hard mask).** Freeze the first `K` queued rows and
  merge from row `K` on — what RTC calls hard masking and REMAC "prefix
  preservation", done by inflating the stale skip. On its own it only
  moves the jump later; LiPo combines it with a linear blend ("freeze during
  the delay window, then blend").
- **Jerk-minimizing post-optimization (LiPo).** Solve a small box-constrained
  QP per chunk: minimize ∫‖q⃛‖² subject to bounded deviation from the
  predicted chunk (looser inside the blending zone, zero during the delay
  window). Client-side and policy-agnostic; needs a QP solver in the loop.
- **ACT temporal ensembling.** Keep every chunk's prediction for each
  future step and average with `w_i = exp(−m·i)` by chunk age. Needs three
  or more overlapping chunks, i.e. requests nearly every step, which the
  server latency does not allow at 30 Hz; with two chunks it degenerates to
  a fixed-weight blend. Both the RTC and the TAS papers report it hurts
  multimodal policies (TAS: 88.2 % vs 33.2 % on Push-T).
- **Output filters.** An EMA or per-step rate limit on the *published*
  action stream smooths everything, not just hand-offs, and adds lag
  everywhere; orthogonal to the merge and easy to add later.
- **Alignment search (VLA-RAIL).** Instead of trusting the step counter for
  the stale skip, search a small window of shifts for the one whose motion
  direction best agrees with the executing plan. Only worth it if the
  observation timestamps are unreliable; this client counts steps exactly.

## 4. Server-side or training-time (out of scope for this client)

- **RTC / IT-RTC** (Physical Intelligence; in LeRobot for π₀, π₀.₅, SmolVLA).
  Guidance inside the flow-matching denoising loop pulls the overlapping
  rows of the new chunk toward the executing chunk's tail (soft prefix
  mask: `LINEAR` / `EXP` / `ONES` / `ZEROS` schedules, `execution_horizon`
  8–12, `max_guidance_weight` ≈ 10 for 10 denoising steps). The client must
  send the previous chunk's leftover and the expected `inference_delay`.
  Strongest smoothing in the literature (SEAM reports RTC at −54 % boundary
  jerk) at ≈1.2× denoising cost (REMAC measures +55–64 ms). The 2026 survey
  finds IT-RTC collapses toward naive beyond ≈8 steps of delay on 50-step
  chunks — borderline for π₀.₅ latency at 30 Hz. openpi in this repo has no
  RTC code; adding it means changing `sample_actions` on the server.
- **SEAM.** Training-free correction after each Euler step of the flow
  (no backward pass): −28 % boundary jerk at 1.01× cost with unchanged
  success; same server-side placement as RTC.
- **TT-RTC.** Fine-tune with simulated inference delay and prefix
  conditioning so the policy itself predicts only the non-stale rows; no
  inference overhead, ≈25 % extra training on Kinetix, degrades on long
  chunks and large delays.
- **REMAC.** Masked action chunking with delay conditioning via LoRA
  (~1.5 % params); prefix-preserved sampling at inference.
- **A2C2.** A small correction head run every control step that adds a
  time-aware residual to the stale chunk. The survey's pick for long-chunk
  VLAs when training a head is acceptable.
- **VLASH.** Roll the proprioceptive state forward with the actions that
  will execute during inference and condition on that future state; needs
  retraining with temporal-offset augmentation.
- **Bidirectional Decoding (BID).** Sample N chunks per observation and pick
  the one most coherent with the previous plan (and most distinct from a
  weaker policy's). Needs batched sampling; the openpi websocket server
  infers one sample per request.
- Others in the same space: DiscreteRTC (discrete-diffusion policies),
  ABPolicy (B-spline action parameterization), Legato / Action ControlNet
  (learned continuation), all requiring a different policy or adapter.

The survey's rule of thumb: IT-RTC if retraining is impossible and delay
is low; TT-RTC if fine-tuning is cheap and chunks are short; A2C2 for
long-chunk VLAs.

## 5. How to compare strategies here

1. Same checkpoint, same `num_episodes`, one run per strategy:
   `./run_eval.sh <name> -- --chunk-merge replace`, `... offset_decay`,
   `... crossfade` (add `--chunk-blend-steps N` to sweep the window).
2. Read `switch_jump_mean/max`, `step_jump_mean/max`, `queue_dry_steps`
   and success from `eval_results/<name>/result.jsonl` / `summary.md`.
3. Expect `switch_jump_*` to drop sharply for `offset_decay` and
   `crossfade` versus `replace`; `offset_decay` should keep success at or
   above `replace` (same reactivity), `crossfade` may trade a little
   reactivity inside its window.
4. Sanity-check a strategy without the sim against a live policy server:
   `python3 fake_obs_client.py --host <ip> --port 8000 --chunk-merge offset_decay`.

## 6. Sources

- Black et al., *Real-Time Execution of Action Chunking Flow Policies*
  (RTC) — https://arxiv.org/abs/2506.07339 ; project page
  https://www.pi.website/research/real_time_chunking ; LeRobot
  implementation and parameters — https://huggingface.co/docs/lerobot/rtc
- LeRobot *Asynchronous Inference* (`aggregate_fn`, `chunk_size_threshold`
  tuning) — https://huggingface.co/docs/lerobot/async
- *Understanding Asynchronous Inference Methods for Vision-Language-Action
  Models* (survey comparing IT-RTC, TT-RTC, VLASH, A2C2) —
  https://arxiv.org/html/2605.08168v1
- *VLA-RAIL: A Real-Time Asynchronous Inference Linker* (client-side
  alignment, polynomial smoothing, dual-quintic stitching) —
  https://arxiv.org/html/2512.24673
- *Hy-Embodied-0.5-VLA* deployment recipe (tangent-aligned cubic Bézier
  stitching) — https://arxiv.org/html/2606.14409
- *LiPo: A Lightweight Post-optimization Framework for Smoothing Action
  Chunks* — https://arxiv.org/html/2506.05165
- *SEAM: Smooth Execution of Action-Chunked Motion* —
  https://arxiv.org/html/2607.04609
- *Real-Time Robot Execution with Masked Action Chunking* (REMAC) —
  https://arxiv.org/html/2601.20130v1
- *Training-Time Action Conditioning for Efficient Real-Time Chunking*
  (TT-RTC) — https://arxiv.org/pdf/2512.05964
- *Leave No Observation Behind: Real-time Correction for VLA Action Chunks*
  (A2C2) — https://arxiv.org/abs/2509.23224
- *VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference* —
  https://arxiv.org/html/2512.01031v2
- *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time
  Sampling* — https://arxiv.org/abs/2408.17355
- *Temporal Action Selection for Action Chunking* (TAS) —
  https://arxiv.org/html/2511.04421v2
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware* (ACT, temporal ensembling) — https://arxiv.org/abs/2304.13705
