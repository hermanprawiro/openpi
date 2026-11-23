# EBIM Task 2 Run Viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only web viewer for the robot client's run folders (`robot_runs/<run>/`), used live by the Munich operator and as a replay tool by us afterwards. Every policy dimension is labelled with its index and a plain name.

**Architecture:** A stdlib `http.server` process reads the recorder's files incrementally (`viewer_data.py`), reconstructs which chunk row produced each executed step (`viewer_align.py`), and serves JSON plus a static, dependency-free page (`viewer_static/`). All data logic lives in tested Python; the JavaScript only fetches and draws. The client's control path is not touched.

**Tech Stack:** Python ≥ 3.10 stdlib (`http.server`, `json`, `re`), numpy, PyYAML (optional), hand-written HTML/CSS/ES-module JS with `<canvas>`, pytest, Docker compose.

**Spec:** `docs/superpowers/specs/2026-11-23-ebim-task2-run-viewer-design.md` — read it first, especially "Facts the viewer is built on" and "Chunk alignment".

## Global Constraints

- Branch `ebim/task2-real-robot-client`. One commit per task. Do not push. Never commit `.gitignore` or `checkpoints/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Client frozen:** no edits to any existing file under `examples/ebim_task2/robot_client/` or `examples/ebim_task2/eval_client/` except adding new files. Release gate: `git diff --diff-filter=MDR --name-only d833586 -- examples/ebim_task2/robot_client examples/ebim_task2/eval_client` prints nothing (only added files).
- No new dependencies (pip or apt). Viewer runtime imports: stdlib, numpy, `robot_client.layout`, `robot_client.labels`, `robot_client.viewer_data`, `robot_client.viewer_align`; `yaml` imported lazily inside `config_summary` in a `try` (missing → config summary `null`). Never import `robot_client.recorder`, `robot_client.config`, `rclpy`, `eval_client` from viewer modules (tests may).
- Offline page: no external URLs anywhere in `viewer_static/` (no CDN, fonts, analytics).
- CLI: `python3 -m robot_client.viewer --runs <dir> [--host 127.0.0.1] [--port 8765]`; defaults exactly `127.0.0.1` and `8765`.
- Read-only: GET only; the server never creates, modifies or deletes files.
- JSON responses never contain `NaN`/`Infinity` (non-finite → `null`).
- Polling only while following a live **run**: episode data (`steps`, `chunks`, `summary`) every 500 ms while the selected episode is live; `/api/runs/<run>` every 2 s while the run is live (new episodes, gate-level events). No timers for finished runs.
- JavaScript is ES2017 + ES modules only (no `?.`, `??`, object spread, class fields), so the `esprima` parse check works; no `node` on this machine.
- Screenshots: Firefox here is a snap with a private `/tmp`; write screenshots under `$HOME/.cache/ebim_viewer_shots/`.
- Step modes are exactly `run, wait, leash, hold, paused`; FREEZE is an episode ending, not a mode.
- Gripper action dims 7/15 are a width **fraction 0–1**; state dims 7/15 are knuckle **rad**.
- Tests: `uv run pytest examples/ebim_task2/robot_client/tests -m "not ros" -q` from the openpi root, and in the image: `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client pytest /app/robot_client/tests`. Servers in tests bind port 0 (host networking: a running viewer may hold 8765).
- `viewer_fixture` is imported as `from viewer_fixture import make_run` (pytest puts `tests/` on `sys.path`, as for `test_runner`). Manual use: `cd examples/ebim_task2/robot_client/tests && PYTHONPATH=../.. uv run python -c "from pathlib import Path; from viewer_fixture import make_run; make_run(Path.home()/'.cache/ebim_viewer_runs', 'run_demo', hold_at=(25, 35))"`.

## Review Focus

1. **Live, mid-write files**: a partial last line in `steps.jsonl`, `chunks.npz` replaced between `stat` and read → no error; the data appears on the next poll. (Tests: Task 2 `test_steps_tail_partial_line`, `test_chunks_cache_retry`.)
2. **Runs with nothing to show**: check-only run, refused run, Enter then instant FREEZE (empty `steps.jsonl`, no `chunks.npz`) → "no data yet", not an error. (Task 2 `test_episode_without_chunks`, Task 4 `test_api_empty_episode`.)
3. **A chunk that never executes** (first chunk fails the start check → FREEZE, whose receipt tick is never written to `steps.jsonl`; chunk received right before a HOLD) → aligned with zero executed rows and its start offset still reported. (Task 3 `test_start_check_freeze`, `test_hold_drops_queue`.)
4. **Hostile paths**: `..`, `%2e%2e`, absolute paths, unknown camera, symlink pointing outside `--runs` → 404. (Task 4 `test_path_traversal_rejected`.)
5. **Viewer started before the run exists** (empty or missing `--runs` dir) → empty list, then the run appears without a restart. (Task 4 `test_runs_dir_appears_later`.)

---

## File Structure

| File | Responsibility |
|---|---|
| `examples/ebim_task2/robot_client/labels.py` | Label table for the 16 munich16 dims (single source for server and page) |
| `examples/ebim_task2/robot_client/viewer_data.py` | Read run folders: list runs, events, config summary, incremental steps, chunks, images; JSON sanitising |
| `examples/ebim_task2/robot_client/viewer_align.py` | Chunk→step alignment, start offsets, clamp flags (pure numpy) |
| `examples/ebim_task2/robot_client/viewer.py` | HTTP server, routing, path validation, `main()` |
| `examples/ebim_task2/robot_client/viewer_static/{index.html,style.css,app.js,plots.js,inspector.js}` | The page |
| `examples/ebim_task2/robot_client/tests/viewer_fixture.py` | Builds run dirs by driving the real `Runner` (via `test_runner.Rig`) into a real `RunRecorder` (tests + manual UI checks) |
| `examples/ebim_task2/robot_client/tests/test_labels.py`, `test_viewer_data.py`, `test_viewer_align.py`, `test_viewer_server.py` | Tests |
| `examples/ebim_task2/compose.robot_client.yml` | New `viewer` service (profile `viewer`) |
| `examples/ebim_task2/README_ROBOT_CLIENT.md`, `examples/ebim_task2/DEPLOY_REAL.md` | Docs |

The Dockerfile already copies `robot_client/` (static files included); no Dockerfile change.

---

### Task 1: Label table

**Files:**
- Create: `examples/ebim_task2/robot_client/labels.py`
- Test: `examples/ebim_task2/robot_client/tests/test_labels.py`

**Interfaces:**
- Produces: `LABELS: tuple[Label, ...]` (len 16); `@dataclass(frozen=True) class Label: index: int; short: str; long: str; side: str ("left"|"right"); kind: str ("joint"|"gripper"); state_unit: str; action_unit: str`; `def display(i: int) -> str` → `"[i] short"`; `def as_json() -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
from robot_client import labels, layout

def test_table_matches_layout():
    assert len(labels.LABELS) == layout.ACTION_DIM == 16
    assert [l.index for l in labels.LABELS] == list(range(16))
    for side, sl in layout.ARM_SLICES.items():
        assert all(labels.LABELS[i].side == side and labels.LABELS[i].kind == "joint" for i in range(sl.start, sl.stop))
    for side, i in layout.GRIPPER_INDEX.items():
        g = labels.LABELS[i]
        assert (g.side, g.kind, g.state_unit, g.action_unit) == (side, "gripper", "rad", "0-1")

def test_display_names():
    assert labels.display(11) == "[11] R J4 elbow"
    assert labels.display(0) == "[0] L J1 base yaw"
    assert labels.display(7) == "[7] L gripper"
    assert labels.display(15) == "[15] R gripper"
    assert "right_fr3v2_joint4" in labels.LABELS[11].long

def test_as_json_round_trip():
    rows = labels.as_json()
    assert rows[8] == {"index": 8, "short": "R J1 base yaw", "long": labels.LABELS[8].long, "side": "right",
                       "kind": "joint", "state_unit": "rad", "action_unit": "rad", "display": "[8] R J1 base yaw"}
```

- [ ] **Step 2: Run** `uv run pytest examples/ebim_task2/robot_client/tests/test_labels.py -q` → FAIL (`ImportError`).
- [ ] **Step 3: Implement `labels.py`.** Joint plain names, in order J1..J7: `base yaw, shoulder, upper-arm roll, elbow, forearm roll, wrist pitch, flange roll`; prefix `L `/`R `. `long` = e.g. `"right FR3 joint 4 (right_fr3v2_joint4), target/measured angle"`; gripper `long` = `"right gripper: state = knuckle angle (rad), action = width fraction 0-1"`. Build the tuple from `layout.ARM_SLICES` / `GRIPPER_INDEX`, not hard-coded indices. Imports only `dataclasses` and `layout`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `Viewer: label table for the munich16 dims`.

---

### Task 2: Run-folder reading + test fixture

**Files:**
- Create: `examples/ebim_task2/robot_client/viewer_data.py`, `examples/ebim_task2/robot_client/tests/viewer_fixture.py`
- Test: `examples/ebim_task2/robot_client/tests/test_viewer_data.py`

**Interfaces:**
- Produces (fixture, test-only): `make_run(out_dir: Path, name: str = "run_test", *, overrides: tuple[str, ...] = (), timeout: float = 6.0, respond=None, ages=None, keys=None, dry_run: bool = False, end: str | None = "operator", nan_t_request: bool = False, check_obs: bool = False, episodes: bool = True, hold_at: tuple[int, int] | None = None) -> Path`.
  - **Drives the real control loop**: builds `test_runner.Rig(overrides, respond=respond, keys=keys, timeout=timeout)` (`robot_client/tests/test_runner.py:151`), replaces `rig.runner.recorder` with a real `RunRecorder(out_dir, name, <config dict>, {"layout": "munich16", "checkpoint": "test/0"}).episode(0)` before `rig.run()`, so `steps.jsonl`, `chunks.npz` and images are exactly what the runner writes. Default `respond` = `lambda n: (3, rig.chunk(tag=0.0), None)` (3-tick latency). `hold_at=(a, b)` sets `rig.ages` so `cam/head` is 2.0 s stale on ticks a..b-1 (→ HOLD, reason from `safety.evaluate`); use `(25, 35)` with the default horizon so the hold lands inside chunk 0's executed rows (ticks 3–32).
  - Writes run-level files the CLI would: `config.yaml` from `rig.cfg.redacted_dict()` (JSON round-tripped like `recorder.py:85`), events `run_start`, `episode_start{episode: 0, dry_run}`, then per `end`: `"operator"`/`"timeout"`/`"freeze"` → `episode_end{ended_by, reason, steps}` + `EpisodeRecorder.finish` (outcome `s`, or `freeze`) + `outcome` + `run_end`; `"interrupted"` → `finish("freeze", "interrupted (Ctrl-C / SIGTERM / SIGHUP)", ...)` + `interrupted` event + `run_end`; `None` → nothing after the steps (a live run).
  - `nan_t_request=True` rewrites chunk 0's `t_request` to NaN in `chunks.npz` after the run. `check_obs=True` writes three 224×224 `check_obs_<camera>.jpg` into the run dir; `episodes=False` skips the episode entirely (a check-only run: config, events `run_start`/`run_end` only if `end`).
- Produces (`viewer_data.py`):
  - `sanitize(obj) -> obj`: numpy → Python, non-finite floats → `None` (recursive).
  - `list_runs(root: Path, now: float | None = None) -> list[dict]` → `[{"name", "kind": "episodes"|"check-obs"|"other", "mtime", "live"}]` newest first; directories only; missing `root` → `[]`. `kind="episodes"` if any `episode_NNN/` dir, `"check-obs"` if any `check_obs_*.jpg`, else `"other"`. `live` = `events.log` has `run_start` and no `run_end`, and the newest mtime of `events.log` and the last episode's `steps.jsonl` is within `LIVE_STALE_S = 7200` s of `now` (the client is silent for minutes at the start gate; a crashed client leaves no `run_end`).
  - `read_events(run_dir: Path) -> list[dict]` (skips unparsable lines).
  - `config_summary(run_dir: Path) -> dict | None` → `{"active_arms", "mode", "execute_horizon", "pin": {int: float}, "safety": {...}}`; `pin` keys are strings on disk (`"7"`) and converted to `int`; `None` if `config.yaml` is missing or `import yaml` fails (lazy import).
  - `episode_list(run_dir: Path, events: list[dict], run_live: bool) -> list[dict]` → `[{"name": "episode_000", "index": 0, "dry_run", "ended_by", "reason", "outcome", "live", "has_result"}]`. `ended_by`/`reason` from `episode_end`; if absent but an `interrupted` event follows this episode's `episode_start` → `ended_by="interrupted"`, `reason` = its note. `outcome` from `result.json` (else the `outcome` event). `live` = `ended_by is None and run_live`.
  - `run_events(events) -> list[dict]`: the gate-level events the page lists: kinds `takeover, freeze_before_start, refused, interrupted, worker_restarted, quit, release_errors, prepare`.
  - `class StepsTail(path: Path)`: `.read() -> list[dict]` returns all complete records so far (cached; reads only appended bytes, holds back a partial last line; resets if the file shrank).
  - `class ChunksCache(path: Path)`: `.read() -> dict | None` with keys `obs_step, t_request, t_receive, latency_steps` (np arrays) and `chunks` `(N, 50, 16)`; `None` if the file is missing or N == 0; reloads only when `(mtime_ns, size)` changes; one retry on `OSError`/`ValueError`/`zipfile.BadZipFile`, then keeps the previous value.
  - `image_index(ep_dir: Path) -> dict[str, list[int]]` (camera → sorted steps); `check_obs_cameras(run_dir: Path) -> list[str]`.

- [ ] **Step 1: Write `viewer_fixture.make_run`** (described above; test support). Sanity: `make_run(tmp)` produces `steps.jsonl` with modes `wait`×3 then `run`, and `chunks.npz` with ≥ 2 chunks.
- [ ] **Step 2: Write the failing tests** in `test_viewer_data.py`:
  - `test_list_runs_kinds_and_order`: an episode run, a check-obs run (`check_obs=True, episodes=False`), a stray file `fake_robot_events.jsonl` → two entries, kinds `episodes`/`check-obs`, file ignored, newest first; missing root → `[]`.
  - `test_live_flag`: `make_run(end=None)` → `live is True` at `now=time.time()` and at `now=time.time()+600` (gate silence), `False` at `now=time.time()+7201`; finished run → `False`.
  - `test_steps_tail_partial_line`: write two full lines plus `'{"step": 2, "mo'`; `read()` → 2 records; append `'de": "run"}\n'` → 3 records, third `{"step": 2, "mode": "run"}`; truncate file to 0 → `read()` == `[]`.
  - `test_chunks_cache_reload_and_retry`: missing → `None`; after `make_run` → shape `(N, 50, 16)`; monkeypatch `np.load` to raise `zipfile.BadZipFile` once → still returns the data (retry); raise always → returns the previous value.
  - `test_episode_without_chunks`: episode dir with empty `steps.jsonl` and no `chunks.npz` → `StepsTail.read() == []`, `ChunksCache.read() is None`, `episode_list` entry present with `ended_by None`.
  - `test_sanitize_nan`: `sanitize({"a": np.float32("nan"), "b": [np.inf, 1.5], "c": np.arange(2)}) == {"a": None, "b": [None, 1.5], "c": [0, 1]}` and `json.dumps(..., allow_nan=False)` succeeds.
  - `test_config_summary`: `mode == "sync"`, `execute_horizon == 30`, `pin == {7: 0.0}`, `active_arms == ["left", "right"]`; with `monkeypatch.setitem(sys.modules, "yaml", None)` → `None`.
  - `test_episode_list_endings`: `end="operator"` → `outcome == "s"`, `ended_by == "operator"`; `end="interrupted"` → `ended_by == "interrupted"`, `outcome == "freeze"`; `end=None` → `live is True`.
  - `test_run_events`: a run with a `freeze_before_start` event and no episode dir → `run_events` returns it.
- [ ] **Step 3: Run** `uv run pytest examples/ebim_task2/robot_client/tests/test_viewer_data.py -q` → FAIL.
- [ ] **Step 4: Implement `viewer_data.py`.**
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `Viewer: incremental run-folder reading + run fixture driven by the real Runner`.

---

### Task 3: Chunk alignment

**Files:**
- Create: `examples/ebim_task2/robot_client/viewer_align.py`
- Test: `examples/ebim_task2/robot_client/tests/test_viewer_align.py`

**Interfaces:**
- Consumes: `StepsTail.read()` records, `ChunksCache.read()` dict (Task 2); `layout`.
- Produces: `align(steps: list[dict], chunks: dict | None, *, mode: str, execute_horizon: int, active_arms: list[str]) -> Alignment` with
  - `Alignment.per_step: list[tuple[int | None, int | None, bool]]` — for each step record in order: `(chunk_k, row, clamped)`; `(None, None, False)` for non-`run` ticks.
  - `Alignment.chunks: list[dict]` — per chunk k: `{"k", "receipt_step": int | None, "receipt_recorded": bool, "receipt_approx": bool, "first": bool | None, "stale_rows": int, "executed": [first_step, last_step] | None, "n_executed": int, "n_clamped": int, "start_offset": {side: float | None}}`.
  - `Alignment.mismatches: int | None` — sync: `run` ticks whose `action` ≠ the assigned row (atol 1e-5) or that have no row; async: `None` (blended merges make value checks meaningless).
  - `Alignment.to_json() -> dict`.

**Rules.** Tick order in `Runner._tick` (`runner.py:286-335`): HOLD entry (queue dropped, epoch bumped) → `_handle_results` (chunk recorded, start check on a first chunk, `queue.on_chunk`) → pop if `run` → `_record` → `plan_step += 1` unless async leash. So `latency_steps[k]` is the receipt tick's `plan_step` minus `obs_step[k]`.
- **Receipt step** of chunk k: `P = obs_step[k] + latency_steps[k]`; the first step record with `plan_step == P` (`receipt_recorded=True`). In async mode leash ticks repeat a `plan_step`, so this may be early by the leash ticks → `receipt_approx = (mode == "async")`. **Unrecorded receipt:** if no record has `plan_step == P` and `P == last_record.plan_step + 1` (the start check FREEZE'd on that tick, `runner.py:366` raises before `_record`), `receipt_step = last_record.step + 1`, `receipt_recorded=False`, and that tick's epoch/measured are taken from the last record. Otherwise `None`.
- `first` = no earlier chunk has its receipt in the same `epoch` (`None` if no receipt).
- `start_offset[side]` = `max |chunks[k, 0, ARM_SLICES[side]] − measured[side]|` at the receipt record (or the last record when unrecorded); `None` without either.
- `clamped` = a `run` tick where some active side's `waypoints[side]` differs from `action[ARM_SLICES[side]]` by > 1e-9 (`clamp_step` returns the target exactly when uncut). `n_clamped` = clamped ticks assigned to chunk k.
- **Sync** — simulate the queue:

```text
queue = deque()
for each step record s (in order):
    if s.mode in ("hold", "paused"): queue.clear()
    for each chunk k with receipt_step == s.step: queue.extend((k, r) for r in range(min(H, 50)))
    if s.mode == "run": (k, r) = queue.popleft() if queue else (None, None); compare s.action with chunks[k, r]
```

- **Async** — mapping by plan step only: a first chunk was loaded with no skip at its receipt (`runner.py:156` passes `next_step=obs_step`), so its row r runs at plan step `P + r`, `stale_rows = 0`; a later chunk skips `latency_steps[k]` rows, so row r runs at plan step `obs_step + r` for `latency ≤ r < latency + H`, `stale_rows = latency`. Each `run` tick is assigned to the most recently received chunk that covers its `plan_step`.

- [ ] **Step 1: Write the failing tests.** Episodes come from the real `Runner` via `test_runner.Rig` with a real `EpisodeRecorder` (reuse `viewer_fixture.make_run`; read back with `StepsTail` / `ChunksCache`). Hand-built records only for `test_receipt_not_yet_recorded` and `test_clamped_flag`.
  - `test_sync_basic`: default fixture (latency 3, H 30) → chunk 0 executes on steps 3–32, chunk 1 on 36–65 (as `test_runner.py:212` asserts for the runner); `mismatches == 0`.
  - `test_sync_leash`: `overrides=("safety.max_track_lag_rad=0.0001",)` with a rig whose measured does not follow → `leash` ticks get `(None, None, False)` and `mismatches == 0`. (If the `Rig` cannot provoke a leash, build these records by hand from a fixture episode by inserting leash ticks and say so in the test docstring.)
  - `test_hold_drops_queue`: `hold_at=(25, 35)` → chunk 0 `n_executed == 22` (steps 3–24); the next chunk is `first is True`; `mismatches == 0`.
  - `test_start_check_freeze`: `respond` returns a chunk offset by 0.5 rad (`rig.chunk(tag=0.5)`) → the episode FREEZEs; chunk 0 has `receipt_recorded is False`, `receipt_step == last_step + 1`, `n_executed == 0`, `start_offset["right"] == pytest.approx(0.5)`, `first is True`.
  - `test_sync_detects_mismatch`: corrupt one recorded `action` → `mismatches == 1`.
  - `test_async_first_and_later_chunks`: `overrides=("inference.mode=async",)` → chunk 0 `stale_rows == 0` and its row 0 lands on its receipt step; chunk 1 `stale_rows == latency_steps[1]`; every `run` tick is assigned; `mismatches is None`.
  - `test_clamped_flag`: a `run` record where `waypoints["right"]` ≠ `action[8:15]` → `clamped True`, `n_clamped == 1`.
  - `test_receipt_not_yet_recorded`: a chunk whose `P` is > last `plan_step + 1` → `receipt_step is None`, no exception.
- [ ] **Step 2: Run** `uv run pytest examples/ebim_task2/robot_client/tests/test_viewer_align.py -q` → FAIL.
- [ ] **Step 3: Implement `viewer_align.py`** per the rules above.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `Viewer: reconstruct chunk-row alignment, start offsets and clamp flags`.

---

### Task 4: HTTP server

**Files:**
- Create: `examples/ebim_task2/robot_client/viewer.py`
- Test: `examples/ebim_task2/robot_client/tests/test_viewer_server.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `make_server(runs: Path, host: str, port: int) -> ThreadingHTTPServer`; `main(argv: list[str] | None = None) -> int`; `if __name__ == "__main__": sys.exit(main())`. API (all JSON via `sanitize` + `json.dumps(allow_nan=False)`, `Cache-Control: no-store`):

| Route | Response |
|---|---|
| `GET /` and `/static/<file>` | files from `viewer_static/` (whitelist: names matching `^[a-z_]+\.(html\|css\|js)$`; `.js` served as `text/javascript`) |
| `GET /api/labels` | `labels.as_json()` |
| `GET /api/runs` | `{"runs": list_runs(...)}` |
| `GET /api/runs/<run>` | `{"name", "config": config_summary, "server": server_metadata.json or {}, "run_events": run_events(...), "episodes": episode_list, "check_obs": [cameras], "live"}` |
| `GET /api/runs/<run>/<ep>/steps?from=<n>` | `{"steps": [record + {"chunk_k", "row", "clamped"}] for records with step ≥ n}` |
| `GET /api/runs/<run>/<ep>/chunks?from=<k>` | `{"chunks": [{"k", "obs_step", "t_request", "t_receive", "latency_ms", "latency_steps", "rows": (50, 16) list}] for k ≥ from}` |
| `GET /api/runs/<run>/<ep>/summary` | `{"episode": episode_list entry, "alignment": Alignment.to_json(), "images": image_index, "result": result.json or null, "stream_stats": ... or null}` |
| `GET /img/<run>/<ep>/<camera>/<step>` | JPEG (`image/jpeg`) |
| `GET /img/<run>/check_obs/<camera>` | JPEG |

- Alignment inputs come from `config_summary`; if it is `None`: `mode="sync"`, `execute_horizon=30`, `active_arms=["left", "right"]`.
- Within one request, read `steps.jsonl` before `chunks.npz` (so chunks are at least as new as steps).
- One `StepsTail`/`ChunksCache` per episode path, kept in a dict guarded by a lock (the server is threaded).
- Path validation: `run` matches `^[A-Za-z0-9][A-Za-z0-9._-]*$` (the CLI's `RUN_NAME_RE`) and is not `.`/`..`; `ep` matches `^episode_\d{3}$`; `camera ∈ {head, wrist_left, wrist_right}`; `step` matches `^\d{1,9}$`; then `resolved.is_relative_to(runs.resolve())`, else 404. Anything else → 404 JSON `{"error": "not found"}`. Non-GET → 405.
- `log_message` is silenced unless `--verbose`. On start, print one line: `Viewer on http://<host>:<port>/ reading <runs> (read-only; Ctrl-C to stop)`. A missing `--runs` dir is not an error.

- [ ] **Step 1: Write the failing tests** (server from `make_server(tmp, "127.0.0.1", 0)` in a thread, `urllib.request` client):
  - `test_api_runs_and_run`: fixture run → `/api/runs` lists it; `/api/runs/run_test` has `config.mode == "sync"`, `server.checkpoint == "test/0"` and one episode.
  - `test_api_steps_incremental`: `/steps?from=0` returns all with `chunk_k`/`row`; `/steps?from=50` starts at step 50.
  - `test_api_chunks_and_summary`: `rows` is 50×16; `summary.alignment.mismatches == 0`; with `make_run(nan_t_request=True)` chunk 0 has `t_request is None` and `latency_ms is None`, and `b"NaN" not in body`.
  - `test_api_empty_episode`: an episode dir with empty files → steps `[]`, chunks `[]`, summary has `alignment.chunks == []`.
  - `test_images`: `/img/run_test/episode_000/head/0` → 200 `image/jpeg`; unknown step → 404.
  - `test_path_traversal_rejected`: `/api/runs/..`, `/api/runs/%2e%2e/episode_000/steps`, `/img/run_test/episode_000/../x/0`, `/img/run_test/episode_000/cam/0`, `/static/../viewer.py`, and a symlinked run dir pointing outside → all 404.
  - `test_runs_dir_appears_later`: start the server on a non-existent dir → `{"runs": []}`; create the run → listed on the next call.
  - `test_static_index`: `/` → 200 `text/html`; no file under `viewer_static/` matches the regex `https?://` (offline page).
  - `test_main_args`: `main(["--help"])` exits 0; parser defaults `host == "127.0.0.1"`, `port == 8765`.
  - `test_no_forbidden_imports`: run `python -c "import robot_client.viewer, sys; print(sorted(m for m in sys.modules if m.startswith(('rclpy', 'eval_client', 'robot_client.recorder', 'robot_client.config'))))"` in a subprocess → prints `[]`.
- [ ] **Step 2: Run** `uv run pytest examples/ebim_task2/robot_client/tests/test_viewer_server.py -q` → FAIL.
- [ ] **Step 3: Implement `viewer.py`** plus a placeholder `viewer_static/index.html` (title "EBIM task2 run viewer") so `test_static_index` has a file.
- [ ] **Step 4: Run** → PASS; also the whole suite `uv run pytest examples/ebim_task2/robot_client/tests -m "not ros" -q` → PASS.
- [ ] **Step 5: Commit** `Viewer: read-only HTTP server and JSON API`.

---

### Task 5: Page shell — navigation, status, images, cursor, live follow

Load the `dataviz` skill before writing any chart or colour code (Tasks 5–7).

**Files:**
- Create/replace: `viewer_static/index.html`, `viewer_static/style.css`, `viewer_static/app.js` (ES module).

**Interfaces:**
- Consumes: the Task 4 API.
- Produces (for `plots.js` / `inspector.js`): `app.js` exports `state = {labels, run, runInfo, ep, steps, chunks, summary, cursor, follow}` and `onChange(fn)` (subscribe to cursor/data changes); `stepIndex(step) -> int` (binary search over `state.steps`).

Behaviour (spec "Page"):
- Header: run picker (kind shown; check-only runs selectable), episode picker, `● LIVE follow` toggle, current mode with colour (`run, wait, leash, hold, paused`) and, if not run, `reason`; `queue`, `in_flight`, `dry_s`; checkpoint id from `server.checkpoint` (fallback `server.config`); `sync H=30` / `async`. If the episode ended with `ended_by == "freeze"`: a red `FREEZE: <reason>` badge; `interrupted` gets an amber `INTERRUPTED` badge.
- Run events: a compact list of `run_events` (time, kind, message/reason) under the header, so gate-level endings (`freeze_before_start`, `refused`, `interrupted`) are visible even when no episode dir exists.
- Banners: `DRY RUN — the arms do not move; measured joints stay flat` when `episode.dry_run`; when the run has no episode: show the check-obs snapshots (that run's, else the newest run with `check_obs`) with the note `Episode view starts after Enter at the start gate. These are the check-obs snapshots from <run>.`
- Images: three 224×224 panels (`head`, `wrist_left`, `wrist_right`), the newest recorded frame ≤ cursor, caption `frame @ step S (cursor C, −d steps)`.
- Cursor: range slider over step records; keys ←/→ ±1, Shift+←/→ ±20, space play/pause (20 steps/s). Scrubbing turns follow off; the follow toggle snaps to the newest step.
- Live: while `follow` and the run is live, refresh `/api/runs/<run>` every 2 s (new episodes are auto-selected; run events update) and, while the selected episode is live, fetch `steps?from=<last+1>`, `chunks?from=<n>`, `summary` every 500 ms. The run list itself refreshes every 5 s only while no run is selected. Finished runs: no timers.
- URL hash `#run=<run>&ep=<ep>&step=<n>` is kept in sync and restored on load (shareable replay links).
- Errors from `fetch` show a small grey "no data yet / server unreachable" note, never an alert.

- [ ] **Step 1: Implement the three files.**
- [ ] **Step 2: Parse check:** `uv run --with esprima python -c "import esprima,sys; [esprima.parseModule(open(f).read()) for f in sys.argv[1:]]" examples/ebim_task2/robot_client/viewer_static/*.js` → no output, exit 0.
- [ ] **Step 3: Smoke check:** create demo runs with the manual fixture command (Global Constraints) under `~/.cache/ebim_viewer_runs` (one finished, one `end=None`), start `uv run python -m robot_client.viewer --runs ~/.cache/ebim_viewer_runs` (cwd `examples/ebim_task2`), and `curl` every endpoint the page uses → 200 with the expected keys. (Visual checks happen once, in Task 7.)
- [ ] **Step 4: Run** the Python suite → PASS (static-file test still green: no external URLs).
- [ ] **Step 5: Commit** `Viewer page: navigation, status header, images, cursor, live follow`.

---

### Task 6: Joint plots, timeline, health strip

**Files:**
- Create: `viewer_static/plots.js`; modify `index.html`, `style.css`, `app.js` (mount points only).

**Interfaces:**
- Consumes: `state`, `onChange`, `stepIndex` (Task 5); per-step `chunk_k`/`row`/`clamped` (Task 4).
- Produces: `export function mountPlots(root)`.

Behaviour (spec "Page"):
- **Timeline** (full width, above the plots): a band coloured by `mode` per step; hovering shows `mode` + `reason`; an end marker with `ended_by: reason` (red for freeze); a cursor line. Clicking sets the cursor.
- **Arm panels**: one column per active arm (from `runInfo.config.active_arms`, default both), right arm first. Per joint a small-multiple canvas titled with `labels[i].display` (tooltip `long`), a shared time axis (seconds from `t`), y in rad, and lines: measured (`measured[side][j]`), commanded (`waypoints[side][j]`), policy action (`action[i]`), the plan of the chunk at the cursor as a faded line from its receipt step over its 50 rows (stale rows greyed). Ticks with `clamped` get a small marker. Cursor line on all plots.
- **Gripper plots**: action (width 0–1) on the left axis, measured knuckle (rad) on a right axis with its own scale and label. If `i` is in `config.pin`: subtitle `policy saw <value> (pinned)`; a missing `knuckle[side]` draws nothing (no zero line).
- **Health strip**: `queue` and `in_flight` (step plot), `dry_s` with a dashed line at `safety.infer_dry_freeze_s`, and chunk latency in ms (`t_receive − t_request`) and in steps at each receipt step.
- Redraw on data/cursor change with `requestAnimationFrame` throttling; drawing must stay < 16 ms for 2400 steps (draw from precomputed typed arrays; recompute those only on new data).

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Parse check** (as in Task 5 Step 2) → exit 0.
- [ ] **Step 3: Smoke check:** parse check passes and the Python suite is green (visual check in Task 7).
- [ ] **Step 4: Commit** `Viewer page: labelled joint plots, mode timeline, health strip`.

---

### Task 7: Chunk inspector and episode result

**Files:**
- Create: `viewer_static/inspector.js`; modify `index.html`, `style.css`, `app.js` (mount points only).

**Interfaces:**
- Consumes: `state`, `onChange` (Task 5); `summary.alignment.chunks`, `summary.result`, `summary.stream_stats` (Task 4).
- Produces: `export function mountInspector(root)`.

Behaviour:
- Shows the chunk at the cursor (that step's `chunk_k`; on a non-`run` tick, the latest chunk received at or before the cursor). Header: `chunk k · requested at plan step obs_step · received step receipt_step · latency X ms / Y steps · first: yes/no · executed rows a–b (n) · stale rows s`.
- Start offset per active arm vs `safety.max_start_offset_rad`: value, threshold, and red if over.
- Alignment line: `alignment: 0 mismatches` (sync) or `N blended` (async); amber if > 0.
- Shows `receipt step ≈` when `receipt_approx`, and `receipt tick not recorded (start check FREEZE)` when `receipt_recorded` is false; `n_clamped` clamped ticks.
- A dimension selector (all 16, by `display` name; default `[11] R J4 elbow`) and one plot of that dimension: the 50 planned rows; measured over the same steps where executed; executed rows highlighted; stale rows greyed; a vertical marker at `execute_horizon`. Clicking a joint plot title in the arm panels selects that dimension here.
- Result panel: `result.json` (outcome, note, stats incl. `ended_by`, `reason`, `infer_ms_mean/max`, `step_clamps`, `holds`), `recorder_errors` (red if non-empty), `stream_stats.json` as a compact table; shows "written after the operator answers the outcome prompt" while absent.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Parse check** → exit 0.
- [ ] **Step 3: Visual check (whole page):** with the Task 5 demo runs plus `make_run(..., 'run_hold', hold_at=(25, 35))`, take `firefox --headless --screenshot ~/.cache/ebim_viewer_shots/page.png --window-size=1600,2400 'http://127.0.0.1:8765/#run=run_hold&ep=episode_000&step=40'` and look at it: mode band shows the hold, `[11] R J4 elbow` shows measured/commanded/plan, the right gripper has two axes, the left reads `policy saw 0.0 (pinned)`, the inspector header matches `curl -s .../summary | python -m json.tool`. Firefox may capture before the fetches finish; if the PNG shows the empty shell, say so in the task report and leave the visual check to the human partner (Task 8 Step 7). Do not add test-only code to the page.
- [ ] **Step 4: Commit** `Viewer page: chunk inspector and episode result`.

---

### Task 8: Compose service, docs, end-to-end, release gate

**Files:**
- Modify: `examples/ebim_task2/compose.robot_client.yml` (add a service; header comment gets a "Viewer" paragraph)
- Modify: `examples/ebim_task2/README_ROBOT_CLIENT.md` (new section 12 appended at the end, after section 11; one link line each in Step 5 and Step 6; Quick reference gets one row)
- Modify: `examples/ebim_task2/DEPLOY_REAL.md` ("After the session": replay command)

**Compose service** (exact):

```yaml
  viewer:
    <<: *robot-client-image
    profiles: [viewer]
    network_mode: host
    volumes:
      - ${OUT_DIR:-./robot_runs}:/out:ro
    command: python3 -m robot_client.viewer --runs /out --host ${VIEWER_HOST:-127.0.0.1} --port ${VIEWER_PORT:-8765}
```

(The entrypoint passes `python3 ...` through unchanged. No `restart:` key.)

**README section** — title `## 12. Optional: watch the run in a browser`, content (plain operator language, same tone as the guide):
- What it is: a page that shows the three camera images the policy gets, every joint and gripper by name, what the policy planned and what was sent to the arms. It only reads the files the client writes; it never moves anything and the client works the same with or without it.
- Start (second terminal, repo root): `docker compose -f examples/ebim_task2/compose.robot_client.yml --profile viewer up -d viewer`, open `http://localhost:8765`. Stop: `docker compose -f examples/ebim_task2/compose.robot_client.yml --profile viewer stop viewer` — **never `down`** (it can remove the running client container too). If the page does not open, `docker compose ... logs viewer` (a taken port 8765: start with `VIEWER_PORT=8766`).
- A browser competes with the client for CPU (section 9). If a second machine is at hand, open the page there (below); on the robot laptop keep only this one tab open.
- It is blank for an episode until you press Enter at `Press Enter to start the policy`; before that it shows the `check-obs` snapshots.
- Browser on another machine: `VIEWER_HOST=0.0.0.0 docker compose ... up -d viewer` and open `http://<laptop ip>:8765`, or keep the default and tunnel: `ssh -L 8765:localhost:8765 <user>@<laptop>`.
- If the page does not load: the client is unaffected; carry on without it.
- DEPLOY_REAL.md "After the session": `cd examples/ebim_task2 && python3 -m robot_client.viewer --runs <unpacked robot_runs>` (numpy needed; pyyaml optional), open `http://localhost:8765`.

- [ ] **Step 1: Edit the compose file and docs.**
- [ ] **Step 2: Build the image:** `docker compose -f examples/ebim_task2/compose.robot_client.yml build` → success.
- [ ] **Step 3: Both suites in the image:** `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client pytest /app/robot_client/tests` and `... pytest /app/eval_client/tests` → all pass.
- [ ] **Step 4: End to end (dev profile, scratch `OUT_DIR`):**
  1. `OUT_DIR=<scratch>/runs docker compose -f examples/ebim_task2/compose.robot_client.yml --profile dev up -d fake-robot stub-server`
  2. `OUT_DIR=<scratch>/runs docker compose ... --profile viewer up -d viewer`; `curl -s localhost:8765/api/runs` → `{"runs": []}` or existing runs.
  3. `OUT_DIR=<scratch>/runs docker compose ... run --rm robot-client check-obs` → the viewer lists it as `check-obs` with three snapshots.
  4. `OUT_DIR=<scratch>/runs docker compose ... run --rm robot-client run --dry-run --episodes 1` (press Enter at the gate, let it run ~20 s, `q`, answer `x`). During the run: `curl -s localhost:8765/api/runs` shows `live: true`; `/summary` shows `alignment.mismatches == 0` and a growing chunk count. Take a headless screenshot during and after (under `~/.cache/ebim_viewer_shots/`).
  4b. **CPU gate:** repeat step 4 twice, once with a Firefox tab following the live episode and once without the viewer; compare `stream_stats.json` (`max_gap_s` and the overrun counts) — the run with the viewer must not be clearly worse (state both numbers in the report).
  4c. **Stop safety:** while a `run --dry-run` container is waiting at the start gate, run `... --profile viewer stop viewer` → the client container is still running (`docker ps`); then quit the client with `q`.
  5. Replay the same folder from the host: `cd examples/ebim_task2 && uv run python -m robot_client.viewer --runs <scratch>/runs --port 8766` → same data.
  6. Tear down both profiles.
- [ ] **Step 5: Release gate:** `git diff --stat d833586 -- examples/ebim_task2/robot_client examples/ebim_task2/eval_client` lists only new files (`labels.py`, `viewer*.py`, `viewer_static/*`, new tests); `grep -rnE "https?://" examples/ebim_task2/robot_client/viewer_static` → no matches; host suite `uv run pytest examples/ebim_task2/robot_client/tests -m "not ros" -q` and `uv run pytest examples/ebim_task2/eval_client/tests -q` → pass.
- [ ] **Step 6: Commit** `Viewer: compose service and operator/deploy docs`.
- [ ] **Step 7: Hand the screenshots to the human partner** with the commit list and the E2E results; do not push.
