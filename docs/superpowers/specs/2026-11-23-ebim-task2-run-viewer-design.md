# EBIM task2 run viewer: design

Date: 2026-11-23. Branch: `ebim/task2-real-robot-client`. Status: draft for approval.
Reviewed by a builder check and by Fable (2026-11-23); their findings are folded in below.

## Intent

The Munich real-robot run is one shot. Two audiences need to see what the policy saw and what it decided:

1. **The operator, live**, during `run --dry-run` and `run`: are the cameras sane, is the plan reasonable, why is
   the robot waiting, holding, or why did it freeze.
2. **Us, afterwards**, replaying the run folder the organizers send back: step through an episode and line up
   images, measured joints, the planned 50-step chunks and what was executed/commanded.

Every policy dimension is shown with **both its index and a plain name**, e.g. `[11] R J4 elbow`.

## Decisions (user)

- Approach A: a separate read-only server that reads the recorder's output folder (live = follow growing files).
- **The client stays frozen.** No change to `recorder.py`, `runner.py`, `streamer.py`, `takeover.py`,
  `safety.py`, `cli.py`, `checks.py`, or any other file on the control path. Consequences accepted: the viewer
  cannot show anything of an episode before the operator presses Enter, and images are the recorder's 2 Hz frames.
- **Ships with the Munich handoff**: a new release after v1 (0e93f2c) that the organizers rebuild. So it must be
  end-to-end tested with the fake robot + stub server, both test suites stay green, and the image still builds.
- Operator browser location is unknown: bind to `127.0.0.1` by default, document `--host 0.0.0.0` and an SSH
  tunnel recipe.

## Goals

- Web page + small Python server reading `<out_dir>/<run>/`. Same page for live and replay.
- Live lag ≤ ~1 s behind the files.
- Fully offline: no CDN, no npm build, no new pip dependencies. Hand-written HTML/CSS/JS, canvas or SVG charts.
- Runs from the robot-client image, and for replay from any Python ≥ 3.10 with numpy (+ pyyaml optional:
  without it the config summary is skipped, not an error).

## Non-goals

- Raw or full-rate camera streams; anything before an episode's Enter (beyond the check-obs snapshots below).
- Controlling anything, connecting to ROS or the policy server, writing into the run dir.
- Dataset / hold-out visualization (the label table can be reused later).

## Anti-goals

- Any change to client behaviour (see Decisions). The viewer must not import `robot_client.recorder`,
  `robot_client.config`, ROS or eval_client; it may import `robot_client.layout` (pure numpy).
- Disturbing the client's 100 Hz streamer: incremental reads only, no polling when not following a live run.
- A missing/partial file producing an error in the page. Missing = "no data yet".
- Making the one-shot operator's job harder: the viewer is optional, clearly skippable, documented after the
  required steps.

## Facts the viewer is built on (recorder contract, verified in code)

Run dir `<out_dir>/<run>/`:
- Every subcommand creates a run dir (`cli.py:501`). Check-only runs have no `events.log` and may hold
  `check_obs_<camera>.jpg` (`checks.py:172`, 224×224 policy image), `check_*` outputs. `out_dir` also holds
  `fake_robot_events.jsonl` in dev. The run list shows only directories, and marks each as `episodes` /
  `check-obs` / `other`.
- `config.yaml` (redacted site config: `active_arms`, `inference.mode`, `inference.execute_horizon`,
  `state.pin`, `safety.*`), `server_metadata.json` (layout, config, checkpoint id).
- `events.log`: run-level only (`run_start`, `episode_start`, `episode_end` with `ended_by`/`reason`/`steps`,
  `outcome`, `takeover`, `freeze_before_start`, `refused`, `worker_restarted`, `quit`, `run_end`, …). **No HOLD
  events.**
- `episode_NNN/` is created **after** the operator's Enter at the start gate (`cli.py:951`).
- `steps.jsonl`, one line per 20 Hz tick, last line may be partial:
  `step, plan_step, t` (seconds since episode t0), `mode ∈ {run, wait, leash, hold, paused}`, `reason` (HOLD /
  pause reason, null in run), `epoch, queue, in_flight, dry_s`, `measured{side: 7}` (rad),
  `knuckle{side}` (raw measured knuckle rad; `left` may be absent), `action` (16-dim, null unless `run`),
  `waypoints{active side: 7}` (step-clamped targets actually commanded, null unless `run`).
  **FREEZE is never a step mode**: the freeze tick is not recorded; the reason is in the `episode_end` event and
  in `result.json` `stats.reason`.
- `chunks.npz` (atomic rewrite per chunk; absent before the first chunk; `(0,0,0)` if none):
  `obs_step` (= `plan_step` of the request), `t_request` (may be NaN), `t_receive`, `latency_steps`
  (= `plan_step` at receipt − `obs_step`), `chunks (N, 50, 16)` float32. A first chunk that fails the start
  check is recorded; a chunk failing `validate_chunk` is not.
- `images/{head,wrist_left,wrist_right}/step_NNNNNN.jpg` every 10 steps (224×224). They are the request's own
  policy images only when a request fell on that tick; otherwise a fresh preprocess of the snapshot.
- `result.json` appears only after the operator answers the outcome + note prompts; `stream_stats.json` with it.
  Episode end for liveness = the `episode_end` event (or `freeze_before_start`).

Gripper slots: state dim 7/15 is a knuckle angle (rad); action dim 7/15 is clipped to [0, 1] before publishing
(`runner.py:403`), i.e. a width fraction. `state.pin` (Munich: `{7: 0.0}`) means the policy saw 0.0 for dim 7
regardless of the measured knuckle.

## Approaches considered

- **A. File-reading viewer (chosen).** No control-path change; one code path for live and replay. Costs:
  nothing before Enter, images at 2 Hz.
- B. Live server inside the client + separate replay viewer. Full-rate images, but a new thread in the safety-
  critical process and two code paths.
- C. Rerun. Little UI code, but a new pinned dependency, logging inside the control loop, generic layout.

## Design

### Labels — `robot_client/labels.py` (pure Python, imports `layout` only)

One entry per dim: `index, short, long, state_unit, action_unit`. Display `[i] short`; `long` in tooltips.

| idx | short | state | action |
|---|---|---|---|
| 0–6 | `L J1 base yaw`, `L J2 shoulder`, `L J3 upper-arm roll`, `L J4 elbow`, `L J5 forearm roll`, `L J6 wrist pitch`, `L J7 flange roll` | rad | rad |
| 7 | `L gripper` | knuckle rad | width 0–1 |
| 8–14 | `R J1 base yaw` … `R J7 flange roll` | rad | rad |
| 15 | `R gripper` | knuckle rad | width 0–1 |

`long` names the ROS joint (`left_fr3v2_joint1` …). The plain names are our call (FR3 has no official ones).
Test: table length = `ACTION_DIM`, arm slices and gripper indices agree with `layout`.

### Chunk alignment (the core of the analysis)

The viewer reconstructs which chunk row produced each executed step, using the recorded columns (never
assuming `step == plan_step`):

- Receipt step of chunk k: the first step record with `plan_step ≥ obs_step[k] + latency_steps[k]` (in sync mode
  leash ticks still advance `plan_step`, in async they freeze it — using the recorded `plan_step` covers both).
- **Sync** (`inference.mode: sync`, Munich): row 0 executes on the first `run` tick at/after receipt; rows
  advance one per `run` tick (`wait`/`leash`/`hold`/`paused` ticks pop nothing); only rows
  `0 … execute_horizon-1` can execute; a HOLD drops the queue (rows after the HOLD are unexecuted).
- **Async**: a first chunk (after start/HOLD) is loaded without skip at its receipt, so row r runs at receipt
  plan step + r; later chunks skip `latency_steps` rows (greyed) and row r maps to `obs_step + r`. Mapping by
  plan step only; executed values may be blended, so "executed" is always the recorded `action`.
- A first chunk that fails the start check is recorded but its receipt tick is not (FREEZE before `_record`):
  the receipt is inferred as the tick after the last record, flagged "not recorded".
- Verification (sync): each executed `action` is compared with the row the reconstruction assigns (float32
  tolerance); the mismatch count is shown in the inspector. This guards the reconstruction instead of trusting
  it. Async shows no mismatch count.

### Server — `robot_client/viewer.py` (stdlib `http.server`, numpy; pyyaml optional)

- `python3 -m robot_client.viewer --runs <dir> [--host 127.0.0.1] [--port 8765]`.
- `ThreadingHTTPServer`. Endpoints (JSON is NaN/Inf-safe: non-finite → null):
  - `GET /api/runs` → `[{name, kind, mtime, live}]`, newest first. `live` = has `run_start`, no `run_end`, and
    `events.log` / the last `steps.jsonl` modified within 2 h (the client is silent at the start gate; a crash
    leaves no `run_end`).
  - `GET /api/runs/<run>` → config summary (`active_arms`, inference mode/execute_horizon, `state.pin`, safety
    thresholds `max_start_offset_rad`, `infer_dry_freeze_s`, `max_step_rad`, …), server metadata, events,
    episodes `[{name, ended_by, reason, outcome, live}]`, check-obs images if present.
  - `GET /api/runs/<run>/<ep>/steps?from=<n>` → records with `step ≥ n`. Cache per file: parsed records + byte
    offset + (mtime, size); a partial last line is held back until complete.
  - `GET /api/runs/<run>/<ep>/chunks?from=<k>` → rows ≥ k plus per-chunk derived: receipt step, executed row
    range, start offset per active arm (max |chunk[0, arm] − measured at receipt|), clamp count. Reload
    `chunks.npz` only when (mtime, size) change; one retry on read error.
  - `GET /api/runs/<run>/<ep>/images` → `{camera: [steps]}`.
  - `GET /api/runs/<run>/<ep>/result` → `result.json` + `stream_stats.json` when present.
  - `GET /img/<run>/<ep>/<camera>/<step>` and `/img/<run>/check_obs/<camera>` → JPEG.
  - `GET /api/labels`.
- Path components validated by strict regex (`run_[A-Za-z0-9_.-]+`-style per the CLI's `RUN_NAME_RE`,
  `episode_\d{3}`, camera ∈ the three names); resolved path must stay under `--runs`. Read-only; GET only.
- Static files from `robot_client/viewer_static/` (`index.html`, `app.js`, `style.css`).
- Logs one line per request at debug level only (quiet terminal).

### Page

```
┌ run ▾ episode ▾  ● LIVE follow │ mode WAIT 0.42s/1.5s │ queue 0 in-flight ● │ ckpt munich_v1/12500 │ sync H=30 ┐
│ [head]  [wrist L]  [wrist R]     frame @ step 1230 (cursor 1236, −6 steps)                                 │
├ timeline: mode band run/wait/leash/hold/paused + HOLD reasons (hover) + episode end marker (FREEZE reason) ─┤
│ ── scrubber ──                                                                                             │
├ Right arm                                        │ Left arm                                                 │
│ [8] R J1 base yaw  measured ─ commanded ─ plan ┄  │ [0] L J1 base yaw ...                                    │
│ ... 7 small multiples, shared time axis           │ ...                                                      │
│ [15] R gripper: action width 0–1 | knuckle rad    │ [7] L gripper: action width; "policy saw 0.0 (pinned)"  │
├ chunk inspector (chunk at cursor): 50 rows vs measured, executed rows highlighted, stale rows greyed,        ┤
│   start offset per arm vs max_start_offset_rad, clamp markers, alignment check                             │
├ health: queue / in_flight / dry_s vs infer_dry_freeze_s · latency per chunk (ms, steps)                    ┤
│ result.json / stream_stats.json / recorder_errors when present                                             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Cursor** = a step. Images: most recent recorded frame ≤ cursor, labelled with its offset in steps. LIVE
  follow sticks to the newest step; scrubbing leaves follow; a button returns.
- **Arm panels**: active arms from `config.yaml` (Munich: both). Per joint: measured (state), commanded
  (`waypoints`), policy action (`action`; differs from commanded where the step clamp fired — mark those
  ticks), and the plan of the chunk at the cursor as a faded line. Gripper: action width 0–1 on the main axis,
  measured knuckle rad on a secondary axis; a pinned state dim is labelled with the pinned value.
- **Dry run** (`run_start.dry_run` / `episode_start.dry_run`): banner "DRY RUN — the arms do not move;
  measured stays flat". The plan/inspector is the useful part there.
- **Before Enter**: if the selected run has no episode yet, show its/the latest check-obs snapshots with the
  note "Episode view starts after Enter at the start gate. These are the check-obs snapshots from <run>."
- **FREEZE**: episode end marker with the reason from `episode_end`; status header shows it in red.
- **Health strip**: `queue`, `in_flight`, `dry_s` with the `infer_dry_freeze_s` threshold line; latency per
  chunk in ms (`t_receive − t_request`) and in steps.
- Live poll: 500 ms for steps/chunks/images/events when following a live episode; otherwise no polling.
- Replay keys: ←/→ one step, Shift+←/→ 20 steps, space play/pause.

### Running it

- **Compose**: service `viewer` under profile `viewer` (not started by `run --rm robot-client`), host network,
  `/out` mounted **read-only**, no restart policy, `command: python3 -m robot_client.viewer --runs /out
  --host ${VIEWER_HOST:-127.0.0.1} --port ${VIEWER_PORT:-8765}`. Start in a second terminal:
  `docker compose -f examples/ebim_task2/compose.robot_client.yml --profile viewer up -d viewer`, open
  `http://localhost:8765`; stop with `... --profile viewer down`.
- **Replay (us)**: `cd examples/ebim_task2 && python3 -m robot_client.viewer --runs <robot_runs dir>`.
- **Docs**:
  - `README_ROBOT_CLIENT.md`: a new short, clearly optional section placed after the step-by-step (not inside
    it), linked from step 5/6 in one line: how to start/stop, that it only reads files and never moves
    anything, that it is blank until Enter, the `VIEWER_HOST=0.0.0.0` / SSH tunnel options, and that the
    client works the same with or without it.
  - `DEPLOY_REAL.md` "After the session": replay command.

### Testing

- Unit (`robot_client/tests/test_viewer.py`, runs in the image and in uv): label table vs layout; incremental
  steps reading with a partial trailing line and file growth; chunks reload and NaN → null; alignment
  reconstruction on synthetic sync and async episodes (incl. wait/leash/hold ticks and a HOLD dropping the
  queue) against `ChunkExecutor` / `AsyncChunkBuffer` behaviour; path traversal rejected; run kinds; API smoke
  test on a run dir produced by the real `RunRecorder` (test-only import), server bound to port 0.
- End to end (before the release): fake robot + stub server, `run --dry-run`, viewer in LIVE follow; then a
  replay of the same folder. Check by eye and record a screenshot.
- Release gate: image builds; both suites green; `git diff d833586 -- <control-path files>` is empty (d833586 = HEAD when this work started).

## Left to the builder

- Chart rendering (canvas vs SVG, colours, ticks), following the dataviz skill.
- Cache structure and exact poll tuning within the lag/CPU limits above.
- Extra config fields in the summary.
