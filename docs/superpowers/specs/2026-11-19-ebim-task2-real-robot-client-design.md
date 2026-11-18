# EBiM Task 2 — real-robot inference & deployment client (Munich diagnostic run)

Date: 2026-11-19 · Branch: `ebim/task2-real-robot-client` (from `ebim/task2-finetune`)

## 1. Intent

We want to run an openpi pi0.5 policy, trained on the organizer-released Munich teleop data, on the
organizers' real Mobile FR3 Duo testbed in Munich. This is a **diagnostic run, not an official
submission**: its purpose is to find out whether a policy trained on real data drives the real robot
sensibly, and where the real-robot path (cameras, latency, control takeover, safety) breaks.

The organizers' operator runs everything. They will not debug our code, so the client must be
self-explanatory, safe to start, configurable to their site without edits to source, and must leave
behind enough logs for us to diagnose a bad run afterwards.

Today we only have a **sim** eval client (`examples/ebim_task2/eval_client/`, Isaac Sim topics,
sim clock, scene resets, 20-dim base+arms+spine layout). Nothing in the repo talks to the real robot.

## 2. Goals, non-goals, anti-goals

**Goals**
1. A **finished training recipe** for the Munich real data, ready to be used as-is for the real full
   fine-tune on another machine: converter, data config, a full-FT `TrainConfig`, norm stats, hold-out
   open-loop eval script, and a runbook with the exact commands (convert → norm stats → train → eval →
   upload). Run the same recipe as a LoRA on this machine's RTX 3090 (10k–15k steps) to prove it end to
   end and to produce a **stand-in checkpoint** the client can serve. The stand-in's quality is
   secondary; the pipeline must be faithful (same layout/preprocessing train vs. deploy).
1b. **Replaceable checkpoint:** swapping the stand-in LoRA checkpoint for the full-FT one (or any later
   step) is a server-side argument change only: no client rebuild, no site-YAML change.
2. A new sibling package `examples/ebim_task2/robot_client/` that runs that policy on the Munich
   testbed: live observations → openpi websocket policy server → smoothed joint/gripper commands.
3. Shipped as a Docker image (ROS 2 Humble, Fast DDS). Source is visible in the image (no protection
   needed). The operator adapts it via a mounted site YAML + env vars, never by editing code.
4. Independent checks the operator can run without robot motion (`check-obs`, `check-infer`),
   plus a bounded motion check (`check-motion`), plus a no-motion end-to-end `run --dry-run`.
5. Defined, conservative behavior on missing inputs, inference stalls, and operator stop.
6. Diagnostic recording: per-episode logs of observations (downsampled), states, chunks, published
   commands, timing, and an operator-entered outcome.
7. A policy-server image/instructions for the organizers' GPU (the server runs there — user
   decision), with our checkpoint mounted.

**Non-goals**
- Base navigation / docking. The operator places the robot at the table as in data collection.
- Running the real full fine-tune ourselves (it runs elsewhere, from the recipe in §6.1), and FSDP /
  multi-node tuning beyond what openpi's `train.py` already offers.
- Official-submission polish (pinned-commit ritual, bilingual README, web viewer UI).
- RTC (real-time chunking) in the server. We use the sim client's sync mode by default and its
  client-side async chunking (latency skip + merge strategies) as an option; RTC can be a follow-up.
- Automatic success detection / scoring. The operator types the outcome.
- Changing the sim eval client's behavior (only small shared-code refactors, sim tests stay green).

**Anti-goals** (failure even if it "works")
- Any path where the arm can receive a target far from its current pose at controller activation,
  or a silent gap in the command stream while the impedance controller is active.
- Fighting another publisher (e.g. a running GELLO teleop) on the arm command topic.
- Joining the robot's ROS graph with a different ROS distro / RMW (the EDL Jazzy container crashed
  the Munich eval PC — issue #39).
- Train/deploy observation mismatch hidden in the client (different crop/resize/state units).

## 3. Constraints and known facts

### 3.1 Munich testbed (from EBiM-Benchmark/submissions #33, #39, #41 evaluator comments, and the
Hajimi Munich profile `sevleete/ebim-task2-offline-submission@c7d3cd79ab`)
- One robot PC `tmr-user@172.16.16.50`; native **ROS 2 Humble**, **Fast DDS** (`rmw_fastrtps_cpp`),
  `ROS_DOMAIN_ID=0`. The evaluation laptop runs our container with host networking.
- Cameras publish raw `sensor_msgs/msg/Image`:
  head `/head_camera/zed_node/rgb/color/rect/image` (~20 Hz),
  wrists `/wrist_camera_{left,right}/camera/color/image_rect_raw` (640×480, ~30 Hz).
  Camera drivers have been seen to die mid-session (evaluator report) → staleness handling matters.
- Arm state: `/<side>/franka_robot_state_broadcaster/measured_joint_states` (JointState,
  names `<side>_fr3v2_joint1..7` per Hajimi's Munich config).
- Arm command: `/<side>/gello/joint_states` (JointState, absolute joint targets) consumed by
  `joint_impedance_controller` under `/<side>/controller_manager`. Hajimi activates the controller only
  after the target topic is already streaming a target equal to the measured pose.
  **Open:** the evaluator described the Munich controller as using "an activation-time reference
  position, relative target changes, and joint-direction signs" — semantics unconfirmed (§9).
- Gripper state: `/<side>/gripper/joint_states`, joint `robotiq_85_left_knuckle_joint` (rad, ~0 open,
  ~0.8 closed). Gripper command: `/<side>/gripper/gripper_client/target_gripper_width_percent`
  (`std_msgs/Float32`, 1.0 = open).
- Arm PTP: action `/<side>/action_server/ptp_motion` (`franka_msgs/action/PTPMotion`); Hajimi
  deactivates the impedance controller before PTP. Spine: action `/franka_spine_node/move_absolute`
  (`franka_spine_msgs/action/MoveAbsolute`). Both message packages come from `franka_ros2` v3.4.1
  (commit `b4164f5`), which EDL built into their Humble image.
- **Provenance.** Confirmed by the evaluator at Munich: the host/IP, Humble + Fast DDS, the three raw
  camera topics (their republish bridges ran *on the evaluation laptop* and saw ~20/30/30 Hz, so raw
  streams reach the laptop at rate), joint/gripper state reception, and PTP execution of a fixed target.
  From Hajimi's Munich profile / the official two-host guide only (not yet confirmed at Munich): the
  arm command topic, the controller name, and the activation order — hence `check-motion` and §9 Q1.
- `franka_msgs/action/PTPMotion.action` and `franka_spine/franka_spine_msgs` (`MoveAbsolute.action`)
  exist in upstream `frankarobotics/franka_ros2` at commit `b4164f5` (verified 2026-11-19), so both can
  be built into our image from upstream.
- Remote inference on an NVIDIA L4 measured ~0.53 s server time / ~0.59 s RTT for a LeRobot pi0.5
  (single sample, evaluator). Expect similar or better for openpi JAX after warmup.

### 3.2 Training data `/home/mislab/datasets/data1/ebim/ebim_task2_realrobotdata/task2_munich`
- LeRobot v2.1, 20 Hz, 238 episodes, **149 valid** (`annotation.human.validity==1`; 89 `REVIEW_FAILED`).
  Valid episode length median 27 s, p90 41 s, max 64 s.
- `action` (17): left 7 joint targets (GELLO follower, absolute rad), left gripper width %, right 7,
  right gripper width %, spine target (constant 434 in every frame). No base.
- `observation.state` (42): per arm 7 measured joints, gripper knuckle rad, 7 external torques,
  6 wrench. No spine state.
- The right arm does the task (per-joint range median ~1.3–1.7 rad); the left arm drifts
  (~0.2–0.3 rad); the left gripper never moves (action always 1.0). Start poses vary across episodes
  (std ~0.2–0.45 rad per joint).
- Loader gotchas: no `task_index` column; episode indices have gaps (14 failed conversions);
  `info.json` claims head 1280×720 but the head video is **336×188 @ ~10 fps**; wrists 640×480
  @ ~29 fps. Video timestamps therefore do not line up 1:1 with the 20 Hz rows.
- Joint names in the dataset are `<side>_fr3_joint*`; the testbed uses `<side>_fr3v2_joint*`
  (cosmetic, mapping in config).

### 3.3 Our stack
- openpi JAX, pinned lerobot 0.1.0 (v2.1 reader). Existing sim converter
  `examples/ebim_task2/convert_task2_data_to_lerobot.py` (pyav decode + `image_tools.resize_with_pad`
  letterbox to 224 at conversion time). Data config `LeRobotEbimTask2DataConfig`, transforms
  `openpi.policies.ebim_task2_policy` (hard-coded `ACTION_DIM = 20`). pi0.5 configs use absolute
  actions (`use_delta_actions=False`).
- Policy server: `scripts/serve_policy.py` (websocket, warmup at startup),
  `scripts/docker/serve_policy.Dockerfile`.
- Reusable sim-client pieces: `inference.py` (reconnecting `PolicyClient`, `InferenceWorker` thread,
  generation-tagged results), `chunking.py` (`AsyncChunkBuffer`, merge strategies, jump metrics —
  but it imports the 20-dim layout constants), `results.py`, the config precedence pattern
  (dataclass < YAML < CLI, `.env` for connection settings).
- This machine: one RTX 3090 24 GB.

## 4. Approaches considered

- **A. Extend `eval_client` with a robot backend.** One codebase, but `episode.py` is built around
  sim time, scene resets and the eval recorder; the refactor would put real-robot safety code into
  sim paths and risk the sim sweeps.
- **B. New sibling package `robot_client/` reusing `inference`/`chunking`/`results`.** ← **picked by
  the user.** Sim client untouched except small shared refactors; real-robot concerns (takeover,
  streaming, safety, site config) live in their own modules with their own tests.
- **C. Port Hajimi's client and swap in our websocket.** Fastest to motion, but left-arm-only 8-dim
  assumptions, no tests, unproven at Munich, and we'd lose our async/merge/metrics.

## 5. Decisions

**User decisions**
- Stay on openpi (JAX) for training and serving.
- Approach B (sibling package).
- Operator = organizers; diagnostic run; Docker image required, source visibility fine; must be
  configurable for site settings.
- Inference server runs on the **organizers' GPU**.
- Train a LoRA policy on this machine for 10k–15k steps if a model is needed (it is).
- **One shot** at Munich: no iteration with the organizers before or during the run.
- The model checkpoint must be **replaceable**; the training recipe and scripts must be finished,
  because they will be used for the real (full fine-tune) training.

**Consequences of one shot** (my reading)
- Full client scope ships in v1: ruckig smoother (with clamp fallback), `relative_to_activation`
  command mode, fake controller-manager integration test, train/deploy parity test. Nothing is
  deferred to "after their first report".
- Every behavior the organizers might need to change on site is a site-YAML key or env var, with a
  documented default; every unconfirmed fact in §9 has a knob plus a check that reveals which
  setting is right (`check-obs`, `check-motion`), and the README tells the operator which knob to turn
  for each failure message.
- A release checklist (§7.1) gates handing over the commit; the stand-in checkpoint ships with it so
  the run can happen even if the full-FT checkpoint isn't ready.

**My calls** (reviewable)
- Policy layout: **16-dim state and 16-dim action**, both arms + both grippers, spine dropped
  (constant). State = `[L 7 measured rad, L knuckle rad, R 7 measured rad, R knuckle rad]`;
  action = `[L 7 target rad, L width %, R 7 target rad, R width %]`, i.e. **raw dataset values, no
  unit conversion** anywhere, so client and converter cannot disagree. Torques/wrench excluded
  (noisy, site-dependent).
- Absolute actions (pi0.5 convention in this repo), action horizon 50 (2.5 s at 20 Hz).
- The client drives a configurable set of arms (`active_arms`), default **both**. A non-active arm is
  held at its measured pose (its policy outputs are logged but not executed).
- Policy grid 20 Hz (dataset fps); command stream 100 Hz.
- Default inference mode **sync** (user doubted `offset_decay`; sim evidence agrees): request a chunk,
  hold the current target while waiting (~0.6 s), execute `execute_horizon` steps, repeat. Sim sweep
  (`STEP_SWEEP_REPORT.md`): sync 33/180, async `blend_favor_old` 27/180, async `replace` 21/180;
  `crossfade`/`offset_decay` 0–1/10 (one weak config only). Sync is also the simplest on hardware: no
  hand-off merge, each chunk starts from the measured pose. Default `execute_horizon` 30 (1.5 s
  open-loop, close to the sim sweep's 50 steps @ 30 Hz = 1.67 s). Async with `blend_favor_old` is the
  site-YAML alternative; the other merges stay selectable but are not recommended in the README.
- Prepare uses the robot's own PTP and spine actions (needs `franka_msgs`, `franka_spine_msgs` built in
  the image), not our streaming path.
- Controller takeover/release via ROS 2 `controller_manager` services from inside the container
  (no SSH into the robot PC).
- Episodes are operator-paced: prepare → Enter → run until timeout or `q` → release → operator types
  outcome → operator resets scene → Enter. Assumes an interactive terminal (`docker run -it`), which
  the evaluator already used for Hajimi's keyboard-driven client.
- Delivery: the organizers build the client image from our public fork (`hermanprawiro/openpi`) at a
  pinned commit on this branch, as the other teams' submissions work; pushing the branch is a separate
  approval from the user. x86_64. Checkpoint via the public HF repo `Ching-Yeh/ebim-task2`
  (`params/` + `assets/`).
- Full-FT recipe defaults copy our existing full-FT configs (`pi05_ebim_task2_fixpos_200_and_v5`):
  pi0.5 from `pi05_base`, `gemma_2b` + `gemma_300m`, batch 32, 30k steps, `ema_decay=0.99`, save every
  5k. 30k × 32 ≈ 12 passes over the ~75k training frames; all overridable from the `train.py` CLI.
- Checkpoint-swap contract: the LoRA and full-FT configs share one data config (same repo id,
  transforms, layout, prompt), so their norm stats and inputs are identical by construction. Both
  configs set `policy_metadata` (`layout: "munich16"`, `action_dim: 16`, `control_hz: 20`,
  `action_horizon: 50`, `train_config`); the client checks this metadata at connect time and refuses
  a mismatched server with a clear message.
- Development while the 3090 is training: client integration runs against a small stub websocket
  policy server (correct metadata and chunk shape; returns the current state plus a small bounded
  sinusoid on one joint), so client work doesn't wait for or compete with training. Integration with
  the real checkpoint happens after training ends.

## 6. Design

### 6.1 Training recipe (finished, reusable) and the stand-in LoRA run (this machine)

Everything in steps 1–3 and 5 is the **recipe** that the real full fine-tune uses unchanged; step 4 is
this machine's run of it. Runbook: `examples/ebim_task2/TRAINING_REAL.md` with the exact commands for
both configs (convert → norm stats → train → hold-out eval → upload to HF), the expected outputs of
each step, disk/VRAM needs, and how to hand the checkpoint to the server.

1. **Converter** `examples/ebim_task2/convert_task2_real_to_lerobot.py` (new; sibling of the sim one):
   - Reads the Munich v2.1 parquet + AV1 videos directly (pyav), keeps `validity==1` episodes.
   - For each 20 Hz row timestamp, takes per camera the latest decoded frame with
     `frame_ts <= row_ts + eps` (head repeats frames at ~10 fps; wrists subsample ~29 fps).
     Decodes sequentially per episode; no reliance on `info.json` shapes/fps.
   - Head stays at native 336×188 before the standard letterbox to 224; wrists 640×480 → letterbox 224
     (same `image_tools.resize_with_pad` as the sim converter and the client).
   - Writes `ebim/task2_munich_real` (v2.1, openpi's lerobot) with `image`, `left_wrist_image`,
     `right_wrist_image`, `state` (16), `actions` (16), task prompt from `tasks.jsonl`.
   - Holds out a fixed list of ~10 valid episodes (seeded) to a separate repo id
     `ebim/task2_munich_real_holdout` for offline checks.
   - Emits `start_pose_stats.json` (mean/std of valid-episode start joint targets per arm) — used as the
     default prepare pose in the site YAML.
   - Deterministic and idempotent (same input → same output, re-run overwrites cleanly); validates
     its output (frame counts per episode, no all-black frames, state/action ranges) and writes a
     `conversion_report.json`.
2. **Transforms/config:** make `EbimTask2Outputs` slice to a configurable `action_dim` (default 20,
   sim unchanged); add an `action_dim` field to `LeRobotEbimTask2DataConfig` passed through. One
   shared data config instance (repo id, transforms, `prompt_from_task=True`, absolute actions,
   `action_dim=16`) used by two `TrainConfig`s:
   - `pi05_ebim_task2_munich_real` (**full FT, the real run**, off this machine): defaults per §5.
   - `pi05_lora_ebim_task2_munich_real` (stand-in, this machine): LoRA from `pi05_base`, batch 16
     (8 if it OOMs on the 3090), 15k steps, save every 2.5k, keep all saved steps.
   Both set the same `policy_metadata` (§5).
3. **Norm stats** via `scripts/compute_norm_stats.py` (per config name; identical data so identical
   stats — the runbook says so and the eval script checks it).
4. **Stand-in run** on the 3090: measure s/step over the first ~100 steps, report ETA; stop at 10k if
   15k is unreasonable. Checkpoints under `checkpoints/pi05_lora_ebim_task2_munich_real/<exp>/`.
5. **Hold-out eval script** `examples/ebim_task2/eval_real_holdout.py` (part of the recipe, run on any
   checkpoint): loads a checkpoint via `policy_config.create_trained_policy`, runs open-loop on the
   hold-out repo, reports per-dim MAE over the first 10/25/50 steps, gripper open/close timing, and
   the first-step offset from measured state (what `max_start_offset_rad` guards at run time). Writes
   `holdout_eval.json` next to the checkpoint. This is the model-quality gate and picks the step to ship.
6. **Upload helper** (documented command or small script) that pushes `params/` + `assets/` of a chosen
   step to `Ching-Yeh/ebim-task2` under `<config>/<exp>/<step>/`, matching the existing repo layout.

### 6.2 Policy server (organizers' GPU)
- Reuse `scripts/docker/serve_policy.Dockerfile` + `scripts/serve_policy.py` with
  `policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=/ckpt`;
  checkpoint (params + assets/norm stats) mounted read-only. Warmup at startup already exists.
- **Swapping the checkpoint** = change `POLICY_CONFIG` (`pi05_lora_…` ↔ `pi05_…`) and the mounted
  directory, restart the server. The server already adds `config` and `checkpoint` to its metadata;
  the client logs both into `server_metadata.json` for every run and validates the §5 layout keys.
- Deliver: a short `SERVER.md` (build, run, expected log lines, how to swap checkpoints, health
  check via `check-infer`).
- Checkpoint distribution: our HF checkpoint repo (layout mirrors openpi's on-disk layout; ship
  `params/` + `assets/`, not `train_state/`).
- GPU memory: pi0.5 inference fits in ~8–10 GB bf16; L4 (24 GB) is fine.

### 6.3 Package layout (`examples/ebim_task2/robot_client/`)

| Module | Responsibility |
|---|---|
| `__main__.py` / `cli.py` | Subcommands: `check-obs`, `check-infer`, `check-motion`, `prepare`, `run` (`--dry-run`), `release` |
| `config.py` | `SiteConfig` dataclasses loaded from YAML; env overrides for server host/port/api key; validation; redacted dump into run dir |
| `layout.py` | State/action index maps built from config (which dims are which arm/gripper); validates against server metadata/chunk width |
| `ros_io.py` | `RobotNode`: subscriptions (raw or compressed images, measured joints, gripper joints) with monotonic arrival stamps + counters; publishers (arm targets, gripper Float32, optional preview topics); publisher-count queries |
| `preprocess.py` | Image pipeline: per-camera optional resize to the training resolution (head → 336×188), then letterbox 224, RGB uint8; gripper/joint extraction with name mapping |
| `takeover.py` | Controller lifecycle via `controller_manager` `list_controllers`/`switch_controller`; pre-checks (no foreign publisher on the command topic, hold target already streaming and equal to measured within tolerance) |
| `prepare.py` | Impedance off → PTP each arm to the configured start pose → spine to configured height → grippers open |
| `streamer.py` | 100 Hz thread: owns the arm command publishers once armed; linear interpolation between 20 Hz waypoints → smoother → command adapter (§6.5) → safety clamps → publish. Always publishes (hold target when idle) |
| `smoother.py` | Jerk/vel/acc-limited smoothing (ruckig if importable, else clamp-based rate limiter) |
| `safety.py` | Joint limits, per-waypoint step clamp, tracking-lag leash, staleness/watchdog evaluation → `RUN` / `HOLD` / `FREEZE` decisions |
| `runner.py` | Episode loop on the 20 Hz wall-clock grid: obs snapshot → `InferenceWorker` → `AsyncChunkBuffer` → waypoints to the streamer; operator keys; episode records |
| `checks.py` | The three checks (§6.7) |
| `recorder.py` | Diagnostic logs (§6.8) |
| `stub_server.py` | Dev tool: minimal websocket policy server speaking the openpi protocol with fake chunks (§5) |
| `fake_robot.py` | Dev tool: replays a Munich episode on the Munich topic names (raw Images, JointStates) and echoes commands back as measured state, so the full client runs on this machine without a robot |
| `tests/` | Unit tests with fakes (no ROS needed), mirroring the sim client's test style |

Shared refactor: `chunking.AsyncChunkBuffer`/merges take `action_dim`, `arm_dims`, `blend_dims`
(defaults = current sim constants); `obs.make_observation`'s letterbox helper reused. The sim client's
tests must pass unchanged.

### 6.4 Runtime state machine (per arm controller, per episode)

```
IDLE ──prepare──▶ PREPARED ──takeover──▶ HOLD ──Enter──▶ RUN ──timeout / q──▶ HOLD ──▶ RELEASED
                                         ▲  │             │
                                         │  └─ p ─────────┤ (pause: clear queue, hold measured)
                                         └── obs stale >hold_s / infer dry ── (auto-resume, replan)
                     any state ── stale >freeze_s / measured dead / Ctrl-C / exception ──▶ FREEZE → RELEASED
```
- **Takeover order:** streamer starts publishing hold target = current measured pose → wait N cycles →
  verify no other publisher on the topic → `switch_controller(activate)` → confirm active and
  measured still ≈ target. Refuse (and say why) on any failed pre-check.
- **Release:** stop policy, hold for 0.5 s, `switch_controller(deactivate)` (arm brakes in place per
  Franka behavior), then stop publishing. Release runs in `finally` and on SIGINT/SIGTERM.
- **Operator gates (plan review):** every prepare, takeover and episode start waits for an explicit
  Enter; a dead or faulted command streamer while a controller is active → FREEZE and release.
- **Left gripper state** is exactly 0.0 in all valid episodes, so the client pins state dim 7 to 0.0
  and doesn't require that stream (`state.pin`, `required: false`). Controller lifecycle can be left to
  the organizers' tooling with `takeover.mode: external` (§9 Q7).
- **Dry-run:** no prepare, no takeover, no gripper commands; the streamer publishes to preview topics
  (`/<side>/robot_client_preview/joint_states`) and gripper commands are suppressed; everything else
  (inference, logging) is identical.

### 6.5 Command path
- Policy action (16) → split per arm via `layout` → only `active_arms` are executed.
- Waypoint queue at 20 Hz. `inference.mode: sync` (default): one blocking request per
  `execute_horizon` (30) steps; the streamer holds the last executed target during the request (a
  normal wait, not a HOLD safety event, but bounded by `infer_dry_freeze_s`). `inference.mode: async`:
  `AsyncChunkBuffer` with latency skip, request when ≤ `trigger_threshold × horizon` remain, merge
  `blend_favor_old`, blend 10.
- Streamer at 100 Hz: `target = lerp(cur_wp, next_wp, u)` → smoother (limits from config, default
  ~50% of FR3 velocity limits) → **command adapter** → clamp → publish.
- **Command adapter** (config `command_mode`):
  - `absolute` (default): publish joint targets as-is.
  - `relative_to_activation`: publish `ref_cmd + sign ⊙ (target − q_at_activation)`, with per-joint
    `joint_signs` and the reference captured at activation. Exists so the operator can switch if the
    organizers confirm the activation-reference semantics (§9) without a code change.
- Grippers: Float32 width % from the action, published only on change > 0.02, clipped to [0, 1].
  Left gripper is effectively constant 1.0 in the data.

### 6.6 Safety (all thresholds in the site YAML)
- Per-joint position limits (FR3 limits minus margin); per-waypoint step clamp (default 0.10 rad);
  smoother vel/acc/jerk limits.
- Tracking leash: if |hold target − measured| > 0.5 rad, stop advancing waypoints until it closes (0.25 would trip in 131/149 training episodes; plan review).
- Observation staleness: any camera or arm/gripper state older than `obs_stale_hold_s` (1.0 s) →
  HOLD (keep publishing current target, no new requests); on recovery drop queued waypoints and
  replan. Older than `obs_stale_freeze_s` (5 s) or measured joints dead > 1.5 s → FREEZE.
- Inference: queue dry → hold current target and keep requesting; dry > `infer_dry_freeze_s`
  (default 5 s) or `FatalInferenceError` → FREEZE.
- Startup: refuse to run if any stream missing, if the server's action width ≠ layout, or if the
  policy's first chunk's first step is > `max_start_offset_rad` (default 0.3) from measured.
- Episode timeout (default 120 s wall; data p90 is 41 s, sync mode runs at ~70 % duty).
- Operator keys: Enter start/resume, `p` pause, `q` end episode, Ctrl-C emergency release.
  Physical E-stop remains the operator's primary safety.

### 6.7 Checks
- `check-obs --seconds 8`: per-stream rate, raw resolution, post-preprocess shape, joint name match,
  gripper value; saves one preprocessed frame per camera to the run dir. No ROS publishers created.
- `check-infer --n 3`: builds observations (live if streams exist, else zeros), reports server
  metadata, chunk shape, latency mean/max, warns if RTT > policy-step budget.
- `check-motion --arm right --delta 0.05`: takeover on one arm, move one configurable joint by
  ±delta via the streamer, confirm measured follows in the expected direction, release. This is
  also the test that disambiguates `command_mode`/`joint_signs`.

### 6.8 Diagnostics recording (mounted `out_dir/<run_name>/`)
- `config.yaml` (redacted), `server_metadata.json`, `events.log` (state transitions, warnings).
- Per episode `ep_XXX/`: `steps.jsonl` (20 Hz: t, state, executed action, queue depth, safety mode),
  `chunks.npz` (obs step, request/receive time, latency steps, full chunk), `stream_stats.json`
  (per-topic rates/gaps), images at 2 Hz as JPEG per camera (preprocessed, what the policy saw), and
  `result.json` with operator outcome (`s`uccess / `f`ail / `x` skip + free-text note).
- Reuse `results.py` jump metrics (`switch_jump_*`, `step_jump_*`).

### 6.9 Site configuration (excerpt of `robot_site.munich.yaml`)
```yaml
ros: {rmw: rmw_fastrtps_cpp, domain_id: 0}
policy: {prompt: "Pick up the thermal pad and place it on the target RAM board", control_hz: 20}
cameras:
  head:        {topic: /head_camera/zed_node/rgb/color/rect/image, type: raw, train_res: [336, 188]}
  wrist_left:  {topic: /wrist_camera_left/camera/color/image_rect_raw, type: raw}
  wrist_right: {topic: /wrist_camera_right/camera/color/image_rect_raw, type: raw}
arms:
  left:
    measured_topic: /left/franka_robot_state_broadcaster/measured_joint_states
    command_topic: /left/gello/joint_states
    joint_names: [left_fr3v2_joint1, ..., left_fr3v2_joint7]
    controller_manager: /left/controller_manager
    controller: joint_impedance_controller
    gripper: {state_topic: /left/gripper/joint_states, joint: robotiq_85_left_knuckle_joint,
              command_topic: /left/gripper/gripper_client/target_gripper_width_percent}
    ptp_action: /left/action_server/ptp_motion
    start_pose: [...]            # from start_pose_stats.json
  right: {...}
active_arms: [left, right]
command_mode: absolute           # or relative_to_activation
joint_signs: {left: [1,1,1,1,1,1,1], right: [1,1,1,1,1,1,1]}
spine: {action: /franka_spine_node/move_absolute, height: 0.434}   # unit to confirm (§9)
inference: {mode: sync, execute_horizon: 30,       # mode: sync | async
            async: {trigger_threshold: 0.5, merge: blend_favor_old, blend_steps: 10}}
safety: {max_step_rad: 0.10, max_track_lag_rad: 0.5, obs_stale_hold_s: 1.0,
         obs_stale_freeze_s: 5.0, infer_dry_freeze_s: 5.0, max_start_offset_rad: 0.3,
         episode_timeout_s: 120, q_margin_rad: 0.05}
smoother: {impl: ruckig, velocity_scale: 0.5}
```
Server host/port/api key come from env (`POLICY_SERVER_HOST/PORT/API_KEY`), as in the sim client.

### 6.10 Docker
- `examples/ebim_task2/Dockerfile.robot_client`: `ros:humble-ros-base` (jammy), `rmw_fastrtps_cpp`,
  `cv_bridge`, `controller_manager_msgs`, `franka_msgs` + `franka_spine_msgs` built from
  `franka_ros2` v3.4.1 (`b4164f5`), python deps (numpy, opencv, pyyaml, ruckig, openpi-client).
  Source copied in (readable). Entry point dispatches the subcommands; the site YAML and `out_dir`
  are mounted; `--network host`; no `--ipc host` (EDL deliberately dropped it).
- `compose.robot_client.yml` with the mounts/env pre-wired; `README_ROBOT_CLIENT.md` for the operator
  (bring-up assumptions, command order: check-obs → check-infer → check-motion → prepare → run
  --dry-run → run, keys, what each failure message means).

### 6.11 Testing strategy
- Unit tests (no ROS): layout, preprocess (incl. head resize→letterbox parity with the converter on
  a dataset frame), command adapter, smoother limits, safety decisions, takeover pre-check logic
  with a fake controller manager, runner with fake node/worker (stale, dry, pause, timeout, FREEZE).
- Sim-client tests unchanged and green after the shared refactor.
- Integration on this machine: policy server (trained checkpoint) + `fake_robot.py` + client in
  Docker → `check-obs`, `check-infer`, `run --dry-run`, and a `run` against the fake robot (fake
  controller manager services) verifying takeover/release ordering and continuous publishing.
- Train/deploy parity: feed hold-out episode frames through the client's preprocess and through the
  converter; assert identical policy inputs; compare open-loop actions.
- Recipe tests: converter on 2–3 real episodes (frame alignment, shapes, validity filter, determinism);
  both `TrainConfig`s construct and their data loaders yield a batch with the expected keys/shapes;
  the stand-in run itself is the end-to-end test of the LoRA path. The full-FT config can't train on
  the 3090, so its first real step happens off-machine; the runbook lists what to check in its first
  log lines.
- Checkpoint swap: serve two different checkpoints (two steps of the stand-in run) back to back
  against the fake robot with the same client image and YAML; a server with a wrong layout (sim
  20-dim config) must be refused at connect time.
- All of the above are required before release (one shot; §7.1).

## 7. Delivery order
1. Converter + transforms/configs (LoRA and full FT) + norm stats → start the LoRA stand-in run
   (long-running; everything else proceeds in parallel). Hold-out eval script, upload helper and
   `TRAINING_REAL.md` next, so the recipe is finished before the client is.
2. Shared refactor of `chunking` (sim tests green).
3. `robot_client` core: config, layout, preprocess, ros_io, streamer/smoother/safety, takeover,
   runner, recorder, checks.
4. `stub_server.py` + `fake_robot.py` integration on this machine while training runs; real-checkpoint
   integration after training ends (the 3090 can't train and serve at once).
5. Dockerfile, compose, operator README, server notes.
6. Offline eval of the stand-in checkpoint; upload it to HF for the organizers.

### 7.1 Release checklist (one shot — all must hold before the commit is handed over)
- All unit, recipe, parity, integration and checkpoint-swap tests green; sim-client tests green.
- Client image builds from a clean clone of the pinned commit on x86_64; `check-obs`, `check-infer`,
  `run --dry-run` and a fake-robot `run` pass from that image.
- Stand-in checkpoint on HF with `holdout_eval.json`; `SERVER.md` serve command verified against it.
- `README_ROBOT_CLIENT.md` maps every refusal/warning message to the site-YAML knob or action that
  fixes it, and lists the §9 questions with the check that answers each on site.
- `TRAINING_REAL.md` walked through end to end with the LoRA config on this machine.

## 8. Left to the builder
- Exact module boundaries/names within the package, thread structure details, log formats.
- Ruckig parameterisation and the clamp fallback.
- Hold-out episode selection seed; checkpoint step to ship (best hold-out MAE).
- Upload helper as a script or as documented `huggingface-cli` commands.
- Wording of operator messages (English).

## 9. Unconfirmed site facts (do not block building; each has a config knob)
One shot means we can't wait for answers: the questions go into the operator README, and the operator
resolves each on site with the named check and knob.
1. Arm command semantics on `/<side>/gello/joint_states` at Munich: absolute targets, or relative to
   an activation-time reference with joint-direction signs? (→ `command_mode`, `joint_signs`;
   `check-motion` verifies.)
2. Will any GELLO/teleop publisher be running during the run? (Client refuses to take over if so.)
3. Spine: units of the dataset value 434 (mm?) and the desired working height. (→ `spine.height`)
4. Head camera: resolution of the live ZED rect topic, and how the dataset's 336×188 head frames
   were produced (plain downscale of the same view?). (→ `cameras.head.train_res`, optional crop.)
5. Joint names on the live topics (`fr3v2` vs `fr3`). (→ `joint_names`)
6. Which GPU hosts the policy server and whether it's reachable from the client laptop on the LAN.
7. Is `controller_manager` `switch_controller` usable from the laptop's DDS graph, or must
   activation go through their own tooling?
