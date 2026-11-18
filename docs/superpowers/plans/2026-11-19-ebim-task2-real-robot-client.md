# EBIM Task 2 Real-Robot Client and Munich Training Recipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A finished openpi training recipe for the Munich real-robot data (LoRA stand-in run here, full fine-tune elsewhere) and a Dockerized ROS 2 Humble client that runs the served policy on the Munich FR3 Duo testbed, operated by the organizers in one shot.

**Architecture:** One shared layout/image contract (`robot_client/layout.py`, `robot_client/preprocess.py`) is imported by the converter, the client and a config cross-check test, so train and deploy cannot drift. The client is a sibling of `eval_client` and reuses its `inference` and (refactored, dimension-parametric) `chunking`. Pure logic (safety, smoothing, command adaptation, takeover ordering, the episode runner) is tested without ROS against protocols. Thin ROS adapters and a fake robot are tested inside the Humble container.

**Tech Stack:** openpi (JAX, pi0.5, lerobot 0.1.0 / dataset v2.1), pyav, openpi-client websocket protocol, ROS 2 Humble + `rmw_fastrtps_cpp` (rclpy, `controller_manager_msgs`, `franka_msgs`, `franka_spine_msgs` from `franka_ros2@b4164f5`), ruckig, PIL, pytest, Docker compose.

**Spec:** `docs/superpowers/specs/2026-11-19-ebim-task2-real-robot-client-design.md` — read it first; this plan argues from it.

## Global Constraints

- Branch `ebim/task2-real-robot-client`. Commit per task; **do not push** (user deferred push). Never commit `.gitignore` changes or `checkpoints/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Sim client behavior unchanged: `uv run pytest examples/ebim_task2/eval_client/tests` stays green after every task.
- Layout: 16-dim state `[L 7 measured rad, L knuckle rad, R 7 measured rad, R knuckle rad]`, 16-dim action `[L 7 target rad, L width %, R 7 target rad, R width %]`, raw dataset values, no unit conversion. Layout id `munich16`.
- Absolute actions (`use_delta_actions=False`), action horizon 50, policy grid 20 Hz, command stream 100 Hz.
- Prompt (exact, no trailing period): `Pick up the thermal pad and place it on the target RAM board`.
- Image pipeline: head resized to `train_res` 336×188 (W×H) only if its live size differs, then `openpi_client.image_tools.resize_with_pad` to 224×224, RGB uint8. Wrists: letterbox only.
- Default inference mode `sync`, `execute_horizon` 30; `async` alternative uses merge `blend_favor_old`, `trigger_threshold` 0.5, `blend_steps` 10.
- Safety defaults: `max_step_rad` 0.10, `max_track_lag_rad` **0.5**, `obs_stale_hold_s` 1.0, `obs_stale_freeze_s` 5.0, `measured_dead_s` 1.5, `infer_dry_freeze_s` 5.0, `max_start_offset_rad` 0.3, `episode_timeout_s` **120**, `q_margin_rad` 0.05, gripper deadband 0.02, smoother `velocity_scale` 0.5. (Leash and timeout changed from the spec's 0.25 / 75 after plan review: in the data, |target − measured| on the right arm exceeds 0.25 rad in 131 of 149 valid episodes, and sync mode's ~70 % duty cycle stretches a p90 41 s episode to ~58 s.)
- **Nothing moves without an operator Enter**: every `prepare`, every takeover and every episode start is gated by an explicit prompt; `run --dry-run` never deactivates/activates controllers, never runs PTP/spine/gripper, never publishes on command topics.
- **The command stream is supervised**: a dead or faulted streamer while any controller we activated is active → FREEZE and release; never a silent gap.
- Left gripper state is **exactly 0.0** in all 149 valid episodes (norm stats q01 = q99 = 0): the client pins state dim 7 to 0.0 by default (`state.pin: {7: 0.0}` in the Munich YAML) and does not require the left gripper state stream (`required: false`).
- Every ROS service/action call has an explicit timeout; every policy-server connect/receive has a timeout.
- Pass through `FASTRTPS_DEFAULT_PROFILES_FILE` (Munich hosts use a `fastdds_udp_only.xml` profile); ship a UDP-only profile in `robot_client/` that the operator can mount/enable.
- Container: `ros:humble-ros-base`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DOMAIN_ID=0`, `--network host`, no `--ipc host`, interactive TTY, x86_64. Source readable in the image. No `cv_bridge` (we decode `sensor_msgs/Image` ourselves; avoids numpy ABI clashes with openpi-client's `numpy>=1.22.4,<2`).
- Operator adapts only via the mounted site YAML, `--set key=value` overrides and env (`POLICY_SERVER_HOST/PORT/API_KEY`); no source edits.
- Swapping checkpoints is server-side only (`POLICY_CONFIG` + mounted dir).
- Tests: unit tests run with `uv run pytest examples/ebim_task2/robot_client/tests -m "not ros"` from the openpi root; ROS tests (marker `ros`) run inside the client container.

## Review Focus

1. **Head camera arrives as `bgra8` (ZED default) or another non-`rgb8` encoding.** Expect correct RGB for `rgb8/bgr8/rgba8/bgra8` and a refusal naming the topic and encoding otherwise. → Task 9, `test_decode_bgra8_and_reject_mono`.
2. **Live JointState with extra, reordered, or differently named joints (`fr3` vs `fr3v2`).** Expect mapping by name; a missing name makes `check-obs` fail and say which joint and which names were seen. → Task 9, `test_state_maps_by_name_and_reports_missing`.
3. **Policy server dies or restarts mid-episode.** Expect HOLD while dry, FREEZE after `infer_dry_freeze_s`, release in `finally`, and the next episode reconnects. → Task 14, `test_server_loss_freezes_then_release`.
4. **A control tick overruns (GC pause, slow snapshot).** Expect the 20 Hz grid to re-anchor without a burst of catch-up waypoints. → Task 14, `test_overrun_reanchors_without_burst`.
5. **Ctrl-C / SIGTERM between streamer start and controller activation, or during PTP; a second Ctrl-C during release.** Expect deactivation only of controllers we activated, publishing stopped, the original exception not masked, and release not aborted by the second signal. → Task 12, `test_interrupt_during_takeover_releases_only_what_we_activated`, Task 17 `test_second_sigint_does_not_abort_release`.

(Plan-review blockers folded in: operator gates and a motion-free dry-run — Tasks 17/19; streamer supervision and adapter activation — Tasks 11/12/14; left-gripper state pin — Tasks 8/9.)

---

## Phase A — Training recipe

### Task 1: Shared layout and image contract

**Files:**
- Create: `examples/ebim_task2/robot_client/__init__.py` (empty), `examples/ebim_task2/robot_client/layout.py`, `examples/ebim_task2/robot_client/preprocess.py` (image part only; Task 9 extends it)
- Create: `examples/ebim_task2/robot_client/tests/conftest.py`, `tests/test_layout.py`, `tests/test_preprocess_image.py`

**Interfaces:**
- Produces (`layout.py`, pure numpy):
  - `LAYOUT_ID = "munich16"`, `STATE_DIM = ACTION_DIM = 16`, `CONTROL_HZ = 20`, `ACTION_HORIZON = 50`, `PROMPT = "Pick up the thermal pad and place it on the target RAM board"`.
  - `ARM_SLICES = {"left": slice(0, 7), "right": slice(8, 15)}`, `GRIPPER_INDEX = {"left": 7, "right": 15}`, `ARM_DIMS = (*range(0, 7), *range(8, 15))`.
  - `DATASET_STATE_INDICES = (*range(0, 7), 7, *range(21, 28), 28)`, `DATASET_ACTION_INDICES = tuple(range(16))` (indices into the Munich 42-dim state / 17-dim action).
  - `POLICY_METADATA: dict` = `{"layout": LAYOUT_ID, "state_dim": 16, "action_dim": 16, "control_hz": 20, "action_horizon": 50, "prompt": PROMPT}`.
  - `class LayoutMismatchError(ValueError)`; `validate_server_metadata(meta: dict, prompt: str) -> None` raises with a message listing each mismatching key (expected vs got); a missing `layout` key reads "server is not a munich16 policy (config=<meta.get('config')>)". A prompt that differs from `meta["prompt"]` is a mismatch.
  - `validate_chunk(chunk) -> np.ndarray` returns the chunk as float64; raises if `ndim != 2`, width **!= 16** (the server's `EbimTask2Outputs(action_dim=16)` already slices; any other width means a wrong-layout server), fewer than 1 row, or any non-finite value.
- Produces (`preprocess.py`): `to_policy_image(frame: np.ndarray, train_res: tuple[int, int] | None, crop: tuple[int, int, int, int] | None = None) -> np.ndarray` — HxWx3 uint8 RGB in, 224×224×3 uint8 out. Optional `crop=(x, y, w, h)` first (site knob for spec §9 Q4; the converter never crops); if `train_res=(W, H)` is given and differs from the frame, PIL `BILINEAR` resize to it; then `image_tools.resize_with_pad(frame, 224, 224)` + `convert_to_uint8`.
- `conftest.py`: same `sys.path` insert as `eval_client/tests/conftest.py`; register marker `ros`; auto-skip `ros` tests when `import rclpy` fails.

- [ ] **Step 1: Write failing tests**

```python
def test_metadata_mismatch_lists_keys():
    meta = dict(layout.POLICY_METADATA, action_dim=20, config="pi05_ebim_task2_fixpos_200_and_v5")
    with pytest.raises(layout.LayoutMismatchError, match="action_dim.*16.*20"):
        layout.validate_server_metadata(meta, layout.PROMPT)

def test_sim_server_without_layout_is_refused():
    with pytest.raises(layout.LayoutMismatchError, match="not a munich16 policy"):
        layout.validate_server_metadata({"config": "pi05_ebim_task2_fixpos_200_and_v5"}, layout.PROMPT)

def test_prompt_mismatch_refused():
    with pytest.raises(layout.LayoutMismatchError, match="prompt"):
        layout.validate_server_metadata(dict(layout.POLICY_METADATA), layout.PROMPT + ".")

def test_validate_chunk_requires_exact_width_and_finite():
    for w in (15, 20, 32):
        with pytest.raises(ValueError): layout.validate_chunk(np.zeros((50, w)))
    bad = np.zeros((50, 16)); bad[3, 2] = np.nan
    with pytest.raises(ValueError): layout.validate_chunk(bad)
    assert layout.validate_chunk(np.zeros((50, 16))).shape == (50, 16)

def test_head_resized_to_train_res_then_letterboxed():
    frame = np.full((720, 1280, 3), 200, np.uint8)
    out = preprocess.to_policy_image(frame, (336, 188))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    native = preprocess.to_policy_image(np.full((188, 336, 3), 200, np.uint8), (336, 188))
    assert np.array_equal(out, native)   # uniform image: identical after both paths

def test_wrist_letterbox_only():
    assert preprocess.to_policy_image(np.zeros((480, 640, 3), np.uint8), None).shape == (224, 224, 3)
```

- [ ] **Step 2:** Run `uv run pytest examples/ebim_task2/robot_client/tests -m "not ros" -v` — expect import errors (FAIL).
- [ ] **Step 3:** Implement `layout.py` and `preprocess.to_policy_image` per the Interfaces block.
- [ ] **Step 4:** Re-run — PASS.
- [ ] **Step 5:** Commit `Add munich16 layout and image contract for the EBIM real-robot client`.

### Task 2: Policy transforms, data config and the two Munich TrainConfigs

**Files:**
- Modify: `src/openpi/policies/ebim_task2_policy.py` (`EbimTask2Outputs`)
- Modify: `src/openpi/training/config.py` (`LeRobotEbimTask2DataConfig` ~L360; new configs after `pi05_ebim_task2_fixpos_200_and_v5` ~L949)
- Create: `src/openpi/policies/ebim_task2_policy_test.py`, `examples/ebim_task2/robot_client/tests/test_train_config_contract.py`

**Interfaces:**
- `EbimTask2Outputs(action_dim: int = ACTION_DIM)` slices `actions[..., :action_dim]`.
- `LeRobotEbimTask2DataConfig.action_dim: int = 20`, passed to `EbimTask2Outputs(action_dim=self.action_dim)`. With `use_delta_actions=True` and `action_dim != 20`, raise `ValueError` (the delta mask is 20-dim layout-specific).
- Module-level in `config.py`: `_MUNICH_REAL_DATA = LeRobotEbimTask2DataConfig(repo_id="ebim/task2_munich_real", base_config=DataConfig(prompt_from_task=True), use_delta_actions=False, action_dim=16)` and `_MUNICH_REAL_METADATA` (literal dict equal to `layout.POLICY_METADATA`; openpi can't import from `examples/`, so Task 2's contract test guards equality).
- `TrainConfig(name="pi05_ebim_task2_munich_real")`: model as `pi05_ebim_task2_fixpos_200_and_v5`, `data=_MUNICH_REAL_DATA`, `pi05_base` weights, `batch_size=32`, `num_train_steps=30_000`, `ema_decay=0.99`, `save_interval=5_000`, `policy_metadata=_MUNICH_REAL_METADATA`.
- `TrainConfig(name="pi05_lora_ebim_task2_munich_real")`: model and `freeze_filter` as `pi05_lora_ebim_task2_fixpos_200_and_v5`, `data=_MUNICH_REAL_DATA`, `batch_size=16`, `num_train_steps=15_000`, `save_interval=2_500`, `keep_period=2_500`, `ema_decay=None`, `policy_metadata=_MUNICH_REAL_METADATA`.

- [ ] **Step 1: Write failing tests**

```python
# src/openpi/policies/ebim_task2_policy_test.py
def test_outputs_default_20_and_configurable_16():
    a = np.zeros((50, 32))
    assert ebim_task2_policy.EbimTask2Outputs()({"actions": a})["actions"].shape == (50, 20)
    assert ebim_task2_policy.EbimTask2Outputs(action_dim=16)({"actions": a})["actions"].shape == (50, 16)

# robot_client/tests/test_train_config_contract.py (skips if `openpi` not importable)
@pytest.mark.parametrize("name", ["pi05_ebim_task2_munich_real", "pi05_lora_ebim_task2_munich_real"])
def test_munich_configs_match_layout(name):
    cfg = _config.get_config(name)
    assert cfg.policy_metadata == layout.POLICY_METADATA
    assert cfg.model.action_horizon == layout.ACTION_HORIZON
    assert cfg.data.action_dim == 16 and cfg.data.use_delta_actions is False
    assert cfg.data.repo_id == "ebim/task2_munich_real"

def test_both_configs_share_data_config():
    assert _config.get_config("pi05_ebim_task2_munich_real").data is _config.get_config("pi05_lora_ebim_task2_munich_real").data

def test_delta_actions_with_16_dims_rejected():  # create() raises
```

- [ ] **Step 2:** Run `uv run pytest src/openpi/policies/ebim_task2_policy_test.py examples/ebim_task2/robot_client/tests/test_train_config_contract.py -v` — FAIL.
- [ ] **Step 3:** Implement per Interfaces. Existing sim configs must not change.
- [ ] **Step 4:** Re-run — PASS. Also `uv run pytest src/openpi/training/config_test.py` if present, and the sim client tests.
- [ ] **Step 5:** Commit `Add Munich real-data pi0.5 full-FT and LoRA configs with 16-dim outputs`.

### Task 3: Munich converter — pure helpers

**Files:**
- Create: `examples/ebim_task2/convert_task2_real_to_lerobot.py` (helpers only in this task), `examples/ebim_task2/tests/test_convert_real.py` (+ `examples/ebim_task2/tests/conftest.py` inserting `examples/ebim_task2` on `sys.path`)

**Interfaces (all pure numpy, importable without lerobot/av at module import — import those inside `main`):**
- `align_frame_indices(frame_ts: np.ndarray, row_ts: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, int]` — for each row, the index of the latest frame with `frame_ts <= row_ts + eps`; rows before the first frame map to 0. Returns `(indices, n_rows_before_first_frame)`.
- `extract_state16(state42: np.ndarray) -> np.ndarray` / `extract_action16(action17: np.ndarray) -> np.ndarray` using `layout.DATASET_*_INDICES`, float32.
- `episode_is_valid(validity: np.ndarray) -> bool` — True iff every value == 1; raise `ValueError` if values are mixed within the episode (report, don't guess).
- `split_holdout(valid_ids: list[int], n: int, seed: int) -> tuple[list[int], list[int]]` — `np.random.default_rng(seed).choice(sorted(ids), n, replace=False)`, both lists sorted.
- `start_pose_stats(first_actions: np.ndarray, episode_ids: list[int]) -> dict` — per arm `{"mean": [7], "std": [7], "medoid": [7], "medoid_episode": int}` plus `"n"`, from per-episode first action rows (16-dim). The medoid is the real episode start minimising the summed L2 distance (per arm) to all other starts; the site YAML uses the **medoid**, not the mean (the mean is 0.147 rad from the nearest real start).
- `validate_episode(images: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray) -> list[str]` — warnings: frame count ≠ rows, any all-black frame (max == 0), non-finite values, |joint| > 3.2 rad, gripper width outside [0, 1].

- [ ] **Step 1: Write failing tests**

```python
def test_align_holds_10fps_head_and_subsamples_29fps_wrist():
    rows = np.arange(6) * 0.05
    head = np.arange(3) * 0.1
    idx, before = convert.align_frame_indices(head, rows)
    assert idx.tolist() == [0, 0, 1, 1, 2, 2] and before == 0
    wrist = np.arange(9) / 29.0
    assert convert.align_frame_indices(wrist, rows)[0].tolist() == [0, 1, 2, 4, 5, 7]

def test_rows_before_first_frame_counted():
    idx, before = convert.align_frame_indices(np.array([0.12, 0.2]), np.array([0.0, 0.05, 0.15]))
    assert idx.tolist() == [0, 0, 0] and before == 2

def test_state_and_action_slices():
    s = np.arange(42, dtype=np.float32)
    assert convert.extract_state16(s).tolist() == [*range(7), 7, *range(21, 28), 28]
    assert convert.extract_action16(np.arange(17.0)).tolist() == list(range(16))

def test_validity_mixed_raises():
    with pytest.raises(ValueError): convert.episode_is_valid(np.array([1, 1, 0]))

def test_start_pose_medoid_is_a_real_start():
    starts = np.zeros((3, 16)); starts[1, 8:15] = 0.1; starts[2, 8:15] = 1.0
    st = convert.start_pose_stats(starts, [4, 7, 9])
    assert st["right"]["medoid_episode"] == 7 and st["right"]["medoid"] == [0.1] * 7

def test_holdout_split_deterministic_and_disjoint():
    a = convert.split_holdout(list(range(100)), 10, 0); b = convert.split_holdout(list(range(100)), 10, 0)
    assert a == b and not set(a[0]) & set(a[1]) and len(a[1]) == 10
```

- [ ] **Step 2:** Run `uv run pytest examples/ebim_task2/tests/test_convert_real.py -v` — FAIL.
- [ ] **Step 3:** Implement the helpers.
- [ ] **Step 4:** Re-run — PASS.
- [ ] **Step 5:** Commit `Add frame alignment and layout helpers for the Munich converter`.

### Task 4: Munich converter `main`, conversion, norm stats, start the stand-in run

**Files:**
- Modify: `examples/ebim_task2/convert_task2_real_to_lerobot.py` (add `main`, tyro CLI like the sim converter)

**Interfaces:**
- `main(raw_dir: str, repo_id: str = "ebim/task2_munich_real", holdout_repo_id: str = "ebim/task2_munich_real_holdout", num_holdout: int = 10, seed: int = 0, max_episodes: int | None = None)`.
- Reads `meta/episodes.jsonl` (gapped indices are fine), per-episode parquet `data/chunk-000/episode_{i:06d}.parquet`, videos `videos/chunk-000/<observation.images.{head,wrist_left,wrist_right}>/episode_{i:06d}.mp4`. Decodes each video fully with pyav (`frame.to_ndarray(format="rgb24")`, `frame.time` for timestamps), ignores `info.json` shapes/fps. Head kept native, then `preprocess.to_policy_image(frame, None)` for every camera (head is already at train_res). Output features as the sim converter (`image`, `left_wrist_image`, `right_wrist_image`, `state` (16,), `actions` (16,)), task string from `meta/tasks.jsonl`.
- Writes `$HF_LEROBOT_HOME/<repo_id>/conversion/conversion_report.json` (per-episode: index, rows, rows_before_first_frame per camera, warnings; skipped invalid ids; holdout ids; seed) and `.../conversion/start_pose_stats.json` (train episodes only). Removes and recreates both output repos (idempotent).

- [ ] **Step 1:** Smoke: `uv run examples/ebim_task2/convert_task2_real_to_lerobot.py --raw-dir /home/mislab/datasets/data1/ebim/ebim_task2_realrobotdata/task2_munich --repo-id ebim/test_munich_smoke --holdout-repo-id ebim/test_munich_smoke_holdout --num-holdout 1 --max-episodes 3`. Expected: 2 train + 1 holdout episodes, report has no warnings except rows-before-first-frame ≤ 2 per camera.
- [ ] **Step 2:** Add `test_converted_smoke_dataset` to `tests/test_convert_real.py` (skip if the smoke repo is absent): loads it with `LeRobotDataset("ebim/test_munich_smoke")` (items come back as CHW float tensors in [0, 1]: convert with `(x * 255).round().byte().permute(1, 2, 0)`), asserts image shape `(224, 224, 3)` after conversion and state/actions `(16,)`, the task string equals `layout.PROMPT`, `state[0]` equals the raw parquet row through `extract_state16`, and — the real parity check — the stored head image of row 0 equals `preprocess.to_policy_image(<raw head frame decoded from the source mp4 at t=0>, (336, 188))` within ±1 per pixel (PNG round-trip). Run it — PASS.
- [ ] **Step 3:** Determinism: re-run the smoke conversion and compare `conversion_report.json` and the first episode's parquet bytes — identical.
- [ ] **Step 4:** Full conversion (no `--max-episodes`). Expected: 149 valid → 139 train + 10 holdout; record totals in the commit message.
- [ ] **Step 5:** `uv run scripts/compute_norm_stats.py --config-name pi05_lora_ebim_task2_munich_real` **once**, then copy `assets/pi05_lora_ebim_task2_munich_real/ebim/task2_munich_real/norm_stats.json` to `assets/pi05_ebim_task2_munich_real/ebim/task2_munich_real/` (`compute_norm_stats.py:48` batches by each config's `batch_size`, so two runs would not match). Assert the copies are byte-identical. Check in `norm_stats.json`: left knuckle (state dim 7) has q01 = q99 = 0 — expected; it is why the client pins that dim. The runbook (Task 6) gives the same compute-once-and-copy commands.
- [ ] **Step 6:** Data-loader check for both configs: `uv run python -c` building `data_loader.create_data_loader(get_config(name), num_batches=1)` (full-FT with `batch_size` overridden to 2) — batch has state `(B, 32)` padded, actions `(B, 50, 32)`.
- [ ] **Step 7:** Start the stand-in run in the background: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lora_ebim_task2_munich_real --exp-name munich_v1 --overwrite`. Measure s/step over the first ~100 steps; if OOM, restart with `--batch-size 8`. Report ETA; stop at 10k if 15k would take > ~36 h.
- [ ] **Step 8:** Commit converter `main` + the smoke test `Add Munich real-data converter with holdout split and conversion report`.

### Task 5: Hold-out open-loop eval script

**Files:**
- Create: `examples/ebim_task2/eval_real_holdout.py`, `examples/ebim_task2/tests/test_eval_real_holdout.py`

**Interfaces:**
- Pure: `chunk_errors(pred: np.ndarray, gt: np.ndarray, horizons=(10, 25, 50)) -> dict[int, np.ndarray]` — per-dim MAE over the first `h` steps, truncating to `len(gt)` when the episode ends early; `gripper_switch_steps(width: np.ndarray, threshold: float = 0.5) -> list[int]`; `gripper_timing_error(pred_w, gt_w) -> float | None` (abs step difference of the first switch; None if either has none); `first_step_offset(pred: np.ndarray, state: np.ndarray) -> float` (max |pred[0, ARM_DIMS] − state[ARM_DIMS]|).
- CLI `main(config_name: str, checkpoint_dir: str, holdout_repo_id: str = "ebim/task2_munich_real_holdout", stride: int = 20, out: str | None = None)`: `policy_config.create_trained_policy(get_config(config_name), checkpoint_dir)`, iterates every `stride`-th frame of each holdout episode, builds the obs dict with keys `observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image`, `observation/state`, `prompt`, and writes `holdout_eval.json` (default `<checkpoint_dir>/holdout_eval.json`) with per-dim MAE per horizon, arm/gripper means, gripper timing stats, first-step offset p50/p95/max, and `norm_stats_sha256` of the checkpoint's assets.

- [ ] **Step 1: Write failing tests**

```python
def test_chunk_errors_truncates_at_episode_end():
    pred = np.ones((50, 16)); gt = np.zeros((20, 16))
    e = ev.chunk_errors(pred, gt)
    assert e[10].tolist() == [1.0] * 16 and e[50].tolist() == [1.0] * 16

def test_gripper_timing():
    gt = np.r_[np.ones(10), np.zeros(10)]; pr = np.r_[np.ones(13), np.zeros(7)]
    assert ev.gripper_timing_error(pr, gt) == 3
    assert ev.gripper_timing_error(np.ones(20), gt) is None

def test_first_step_offset_ignores_grippers():
    pred = np.zeros((50, 16)); pred[0, 7] = 1.0; pred[0, 9] = 0.2
    assert ev.first_step_offset(pred, np.zeros(16)) == pytest.approx(0.2)
```

- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** When the stand-in run has its 2.5k checkpoint (GPU contention: run eval only between checkpoints if VRAM allows, otherwise after training), run `uv run examples/ebim_task2/eval_real_holdout.py --config-name pi05_lora_ebim_task2_munich_real --checkpoint-dir checkpoints/pi05_lora_ebim_task2_munich_real/munich_v1/2500` — JSON written, arm MAE@10 finite.
- [ ] **Step 6:** Commit `Add open-loop holdout eval for Munich checkpoints`.

### Task 6: Upload helper, policy-server launcher and training runbook

**Files:**
- Create: `examples/ebim_task2/upload_checkpoint.sh`, `examples/ebim_task2/TRAINING_REAL.md`, `examples/ebim_task2/compose.policy_server.yml`, `examples/ebim_task2/serve_munich.sh`

**Server launcher (the existing `serve_policy.Dockerfile` has no `POLICY_CONFIG` and ships no source — it runs `uv run scripts/serve_policy.py $SERVER_ARGS` against a bind-mounted repo):**
- `compose.policy_server.yml`: builds `scripts/docker/serve_policy.Dockerfile`, bind-mounts the repo at `/app` (as `scripts/docker/compose.yml` does), mounts `${CKPT_DIR}:/ckpt:ro` and `${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets` (env `OPENPI_DATA_HOME=/openpi_assets`), GPU reservation, `network_mode: host`, `SERVER_ARGS="--port ${PORT:-8000} policy:checkpoint --policy.config=${POLICY_CONFIG} --policy.dir=/ckpt"`.
- `serve_munich.sh <config> <ckpt_dir> [port]` wraps it and checks `<ckpt_dir>/params` and `<ckpt_dir>/assets` exist first.
- The PaliGemma tokenizer is downloaded on first start into `OPENPI_DATA_HOME`; `SERVER.md` (Task 20) gives a one-line pre-fetch for GPU hosts without internet. Verify by starting the server once with the cache pre-filled and the network disabled for the container (`--network none` variant of the compose run) up to "Warmup done" (Task 21).

**Interfaces:**
- `upload_checkpoint.sh <config> <exp> <step> [--repo Ching-Yeh/ebim-task2]`: refuses if `params/` or `assets/` is missing or `holdout_eval.json` is absent; runs `huggingface-cli upload <repo> checkpoints/<config>/<exp>/<step>/params <config>/<exp>/<step>/params` and the same for `assets` and `holdout_eval.json`; never uploads `train_state/`. Prints the server command for that step.
- `TRAINING_REAL.md` sections: prerequisites (disk, VRAM: LoRA fits 24 GB at batch 16; full FT needs multi-GPU/≥80 GB-class, use `--fsdp-devices`), 1 convert (exact command, expected counts 139/10), 2 norm stats (both configs, identical), 3 train (full FT command, LoRA command, overrides), 4 what to check in the first log lines (loss finite and falling within 500 steps, data shapes, s/step), 5 hold-out eval + choosing a step (lowest arm MAE@25 with first-step offset p95 < `max_start_offset_rad` 0.3), 6 upload, 7 serve and swap (`POLICY_CONFIG`, mount), 8 troubleshooting (OOM, stale norm stats, prompt mismatch refusal).

- [ ] **Step 1:** `bash -n examples/ebim_task2/upload_checkpoint.sh`; run it against a directory without `holdout_eval.json` — exits non-zero with the reason.
- [ ] **Step 2:** Dry-walk the runbook: every command in it was run verbatim in Tasks 4–5 (copy them from there); fix drift.
- [ ] **Step 3:** `docker compose -f examples/ebim_task2/compose.policy_server.yml config` renders with `POLICY_CONFIG`/`CKPT_DIR` set; full serve test happens in Task 21 (GPU busy training).
- [ ] **Step 4:** Commit `Add Munich training runbook, server launcher and checkpoint upload helper`.

---

## Phase B — Shared refactor

### Task 7: Dimension-parametric chunking

**Files:**
- Modify: `examples/ebim_task2/eval_client/chunking.py`
- Modify: `examples/ebim_task2/eval_client/tests/test_chunking.py` (add tests only)

**Interfaces:**
- `ChunkExecutor(replan_steps: int, *, action_dim: int = ACTION_DIM)`.
- `AsyncChunkBuffer(execute_horizon, trigger_threshold, merge="replace", blend_steps=10, *, action_dim: int = ACTION_DIM, arm_dims: tuple[int, ...] = ARM_ACTION_DIMS, blend_dims: tuple[int, ...] = BLEND_ACTION_DIMS)`.
- Merge signature becomes `merge(queue, fresh, last_action, blend_steps, blend_idx: np.ndarray)`; `_mix(old, new, w_old, w_new, blend_idx)`; `_validate_chunk(chunk, action_dim)`. Module globals `_ARM`/`_BLEND` removed in favor of instance fields. Existing call sites (`episode.py`) unchanged because of the defaults.

- [ ] **Step 1: Write failing tests**

```python
def test_munich16_blend_leaves_grippers_fresh():
    buf = chunking.AsyncChunkBuffer(30, 0.5, "blend_favor_old", action_dim=16,
                                    arm_dims=ARM16, blend_dims=ARM16)
    buf.on_chunk(0, np.zeros((50, 16)), 0)
    fresh = np.ones((50, 16)); buf.on_chunk(0, fresh, 0)
    row = buf.pop()
    assert row[7] == 1.0 and row[15] == 1.0          # grippers take fresh
    assert row[0] == pytest.approx(0.3)               # 0.7*old + 0.3*new
    assert row.shape == (16,)

def test_executor_16_dims():
    ex = chunking.ChunkExecutor(30, action_dim=16); ex.load(np.zeros((50, 32)))
    assert ex.next_action().shape == (16,)
```
(`ARM16 = (*range(0, 7), *range(8, 15))`.)

- [ ] **Step 2:** Run `uv run pytest examples/ebim_task2/eval_client/tests -v` — new tests FAIL, old PASS.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Whole sim suite PASS unchanged.
- [ ] **Step 5:** Commit `Make eval_client chunking dimension-parametric for the robot client`.

---

## Phase C — Robot client

### Task 8: Site configuration

**Files:**
- Create: `examples/ebim_task2/robot_client/config.py`, `examples/ebim_task2/robot_client/robot_site.munich.yaml`, `tests/test_config.py`

**Interfaces:**
- Frozen dataclasses: `CameraConfig(topic: str, type: Literal["raw", "compressed"] = "raw", train_res: tuple[int, int] | None = None, crop: tuple[int, int, int, int] | None = None)`; `GripperConfig(state_topic, joint, command_topic, required: bool = True)`; `StateConfig(pin: dict[int, float] = {})` (state dims replaced by a constant before inference; Munich YAML: `{7: 0.0}`); `TakeoverConfig(mode: Literal["switch", "external"] = "switch", discovery_wait_s=3.0, warm_cycles=20, tol_rad=0.02, service_timeout_s=5.0, hold_inactive_arms=True)` (`external`: the organizers activate/deactivate the controller with their own tooling; we stream hold, wait for `active`, never call switch — spec §9 Q7); `ArmConfig(measured_topic, command_topic, joint_names: tuple[str, ...], controller_manager, controller, gripper: GripperConfig, ptp_action, start_pose: tuple[float, ...], command_joint_names: tuple[str, ...] | None = None)`; `SpineConfig(action, height, velocity=0.1, acceleration=0.2, deceleration=0.2, enabled=True, get_position_service="/franka_spine_node/get_position", max_delta_m=0.2)` (`height` validated in [0.0, 1.0] m so a value like 434 is caught as a unit error); `AsyncConfig(trigger_threshold=0.5, merge="blend_favor_old", blend_steps=10)`; `InferenceConfig(mode: Literal["sync", "async"] = "sync", execute_horizon=30, async_cfg: AsyncConfig)` (YAML key `async`); `SafetyConfig` with the Global-Constraints defaults plus `gripper_deadband=0.02`, `joint_limits: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]` defaulting to FR3 `q_min=(-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)`, `q_max=(2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)` for both sides; `SmootherConfig(impl: Literal["ruckig", "clamp"] = "ruckig", velocity_scale=0.5, v_max=(2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26), a_max=(10.0,)*7, j_max=(5000.0,)*7)`; `PrepareConfig(ptp_max_joint_velocities=(0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4), ptp_goal_tolerance=0.01, timeout_s=60)`; `ServerConfig(host, port=8000, api_key=None, connect_timeout_s=10.0, infer_timeout_s=10.0)` (host/port/key from env only; timeouts from YAML `server:`).
- `SiteConfig(ros: RosConfig(rmw="rmw_fastrtps_cpp", domain_id=0), policy: PolicyConfig(prompt, control_hz=20), cameras: dict[str, CameraConfig], arms: dict[str, ArmConfig], active_arms: tuple[str, ...] = ("left", "right"), command_mode: Literal["absolute", "relative_to_activation"] = "absolute", joint_signs: dict[str, tuple[int, ...]], spine, inference, safety, smoother, prepare, stream_hz=100, out_dir="/out", server)`.
- `load_site_config(path: str | Path, overrides: Sequence[str] = (), env: Mapping[str, str] = os.environ) -> SiteConfig`; overrides are `dotted.key=yaml_value`. Errors are `ConfigError` with *all* problems listed (unknown key with its dotted path; camera keys must be exactly `head/wrist_left/wrist_right`; arms exactly `left/right`; `active_arms` ⊆ arms and non-empty; 7 joint names, 7 start-pose values, 7 signs each ±1; `execute_horizon` in [1, 50]; `policy.prompt` must equal `layout.PROMPT` — an error, since the server check would refuse anyway, with a message naming `policy.prompt`).
- `SiteConfig.redacted_dict() -> dict` (api key → `"***"`); `check_ros_env(cfg, env) -> list[str]` (mismatch between YAML `ros` and `RMW_IMPLEMENTATION`/`ROS_DOMAIN_ID`).
- `SiteConfig` also carries `state: StateConfig`, `takeover: TakeoverConfig`.
- `robot_site.munich.yaml`: §6.9 values plus `state: {pin: {7: 0.0}}`, left gripper `required: false`; `start_pose` = the per-arm **medoid** in `start_pose_stats.json` from Task 4 Step 4, rounded to 3 decimals; `spine.height: 0.434` (dataset value 434 read as mm; Hajimi's Munich profile uses metres, e.g. 0.4).

- [ ] **Step 1: Write failing tests** — `test_munich_yaml_loads_with_defaults` (mode sync, horizon 30, merge blend_favor_old), `test_override_dotted_key` (`safety.max_step_rad=0.05`), `test_unknown_key_named` (`safety.max_stp_rad` → message contains `safety.max_stp_rad`), `test_all_errors_reported_together` (two bad fields → both in message), `test_env_server_and_redaction`, `test_ros_env_mismatch_reported`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `Add robot client site config and Munich profile`.

### Task 9: Observation preprocessing

**Files:**
- Modify: `examples/ebim_task2/robot_client/preprocess.py`; Create `tests/test_preprocess.py`

**Interfaces:**
- `decode_raw_image(data: bytes, height: int, width: int, step: int, encoding: str) -> np.ndarray` — supports `rgb8`, `bgr8`, `rgba8`, `bgra8` (step-aware); else `UnsupportedEncodingError(encoding)`.
- `decode_compressed_image(data: bytes) -> np.ndarray` — lazy `cv2.imdecode`, BGR→RGB.
- `@dataclass Snapshot(images: dict[str, np.ndarray], arm_q: dict[str, np.ndarray], knuckle: dict[str, float], ages: dict[str, float], t_mono: float)`; age keys are stream names `cam/<name>`, `arm/<side>`, `grip/<side>`.
- `map_joints(names: Sequence[str], positions: Sequence[float], wanted: Sequence[str]) -> np.ndarray` raising `MissingJointError(missing, seen)`.
- `build_state(snap: Snapshot, pin: Mapping[int, float]) -> np.ndarray` (16, float32, layout order; pinned dims overwritten; a not-required gripper with no sample uses its pin value, and is an error if it has no pin).
- `state_range_report(state: np.ndarray, norm_stats: dict | None) -> list[str]` — for `check-obs`: per dim, flags values outside the training [q01, q99] widened by 10 % of the range (norm stats read from a path given on the CLI, optional).
- `build_observation(snap: Snapshot, cameras: dict[str, CameraConfig], prompt: str) -> dict` — keys as `EbimTask2Inputs` expects (`observation/image` ← head, `observation/left_wrist_image`, `observation/right_wrist_image`, `observation/state`, `prompt`), images through `to_policy_image(img, cfg.train_res)`.

- [ ] **Step 1: Write failing tests** — `test_decode_bgra8_and_reject_mono` (2×2 bgra8 with step padding → expected RGB; `mono8` raises with "mono8" in message), `test_state_maps_by_name_and_reports_missing` (reordered names with extra joints → correct order; missing `right_fr3v2_joint3` → message lists missing and seen names), `test_build_observation_keys_and_shapes`, `test_pin_overrides_left_knuckle` (live knuckle 0.003 → state[7] == 0.0), `test_missing_optional_gripper_uses_pin`, `test_state_range_report_flags_out_of_range`, `test_crop_applied_before_resize`. (The real converter-vs-client image parity test is Task 4 Step 2; it needs the dataset.)
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add robot client observation preprocessing`.

### Task 10: Safety evaluation and smoothing

**Files:**
- Create: `robot_client/safety.py`, `robot_client/smoother.py`, `tests/test_safety.py`, `tests/test_smoother.py`

**Interfaces:**
- `class Mode(Enum): RUN, HOLD, FREEZE`.
- `evaluate(ages: dict[str, float], infer_dry_s: float, fatal_inference: bool, cfg: SafetyConfig) -> tuple[Mode, str]` — FREEZE if fatal, any `arm/*` or `grip/*` age > `measured_dead_s` (arm only) / `obs_stale_freeze_s`, any camera age > `obs_stale_freeze_s`, or `infer_dry_s > infer_dry_freeze_s`; HOLD if any age > `obs_stale_hold_s`; else RUN. Reason names the worst stream and its age. A stream never seen has age `inf`.
- `clamp_step(prev: np.ndarray, target: np.ndarray, max_step: float) -> tuple[np.ndarray, bool]` (per-joint clamp, flag if clamped).
- `clamp_limits(q, q_min, q_max, margin) -> tuple[np.ndarray, bool]`.
- `lag_exceeded(cmd, measured, max_lag) -> bool`.
- `start_offset(first_target, measured) -> float`.
- `smoother.Smoother` protocol: `reset(q: np.ndarray) -> None`, `step(target: np.ndarray, target_vel: np.ndarray) -> np.ndarray`, `stop() -> None` (subsequent steps decelerate to zero velocity at the current position — used for hold). `RateLimiter(v_max, a_max, dt)`; `RuckigSmoother(v_max, a_max, j_max, dt)` (ruckig `Ruckig(7, dt)` online; `target_position = target`, `target_velocity = clip(target_vel, ±v_max)` — the lerp segment velocity, so a moving target doesn't stop-go; result codes other than `Working`/`Finished` raise `SmootherError`). Inputs are already limit-clamped (clamping happens before the smoother so its internal state never diverges from what is published). `make_smoother(cfg: SmootherConfig, dt: float) -> Smoother` scales `v_max` by `velocity_scale` (and a/j by the same factor); falls back to `RateLimiter` with a logged warning if `impl == "ruckig"` and ruckig is not importable.

- [ ] **Step 1: Write failing tests** — `test_freeze_on_dead_arm_before_hold_on_camera` (arm age 2.0, cam age 1.2 → FREEZE, reason mentions `arm/right`), `test_hold_then_run`, `test_never_seen_stream_is_freeze`, `test_clamp_step_flags`, `test_limits_with_margin`, `test_rate_limiter_respects_v_and_a` (per-step |Δq| ≤ v·dt + 1e-9 and |Δv| ≤ a·dt over a 1 rad step; converges within 5 s), `test_ruckig_respects_limits` (`pytest.importorskip("ruckig")`; runs in the container), `test_tracks_moving_target_without_stopgo` (ramp target at 0.2 rad/s with matching `target_vel` → steady-state velocity within 5 % of 0.2 and never sign-flips), `test_stop_decelerates_without_return` (after `stop()` at speed, position is monotone until velocity reaches 0), `test_make_smoother_fallback` (monkeypatch import failure → `RateLimiter`).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add robot client safety checks and joint smoothers`.

### Task 11: Command adapter and 100 Hz streamer

**Files:**
- Create: `robot_client/command.py`, `robot_client/streamer.py`, `tests/test_command.py`, `tests/test_streamer.py`

**Interfaces:**
- `CommandAdapter(mode: str, signs: np.ndarray)`: `activate(q_activation: np.ndarray, ref_cmd: np.ndarray)`, `deactivate()`, `__call__(target) -> np.ndarray`; `absolute` → identity; `relative_to_activation` → identity until `activate`, then `ref_cmd + signs * (target - q_activation)`. `takeover` calls `activate(q_activation=measured, ref_cmd=streamer.last_command(side))` immediately **before** `cm.switch` (in `external` mode: when the controller is first seen `active`); `release` calls `deactivate()` after deactivation.
- `Streamer(sides: Sequence[str], publish: Callable[[str, np.ndarray], None], measured: Callable[[str], np.ndarray], smoother_factory: Callable[[], Smoother], adapters: dict[str, CommandAdapter], limits: dict[str, tuple[np.ndarray, np.ndarray]], safety: SafetyConfig, hz: float = 100.0, waypoint_dt: float = 0.05, clock=time.monotonic)`.
  - `start(hold: dict[str, np.ndarray])` seeds smoothers and starts the thread; `stop()` joins it (publishing stops only here).
  - `set_waypoint(side, q: np.ndarray)` — ramps linearly from the current interpolated position to `q` over `waypoint_dt`.
  - `hold(side=None)` — no further ramp; smoother `stop()` (decelerate in place, no overshoot-and-return).
  - `step(now: float) -> dict[str, np.ndarray]` — pure-ish one tick: lerp (+ segment velocity) → if `lag_exceeded` vs `measured(side)`, freeze the ramp parameter and set `leash_engaged(side)` → `clamp_limits` → smoother → adapter → publish. Thread calls it at `hz` with absolute-deadline sleeps.
  - **Supervision:** the thread wraps each `step` in try/except; on exception it republishes each side's last command, sets `fault: str | None` (first exception text) and keeps republishing the last command (it does not die). `healthy() -> bool` is `fault is None and thread alive and (now - last_publish) < 0.05 s` for every side. Runner, Session and the pre-Enter wait poll `healthy()`; unhealthy → FREEZE → release.
  - `leash_engaged(side) -> bool` — read by the Runner, which stops consuming actions for that arm while it is set (spec: "stop advancing waypoints").
  - Counters for the recorder: `cycles`, `max_gap_s` (largest interval between publishes per side), `leash_events`, `limit_clamps`.
  - `cycles_since_start(side) -> int`, `last_command(side) -> np.ndarray`.

- [ ] **Step 1: Write failing tests** — `test_relative_mode_math` (signs −1 on joint 2), `test_streamer_publishes_hold_before_any_waypoint` (fake clock, 10 `step`s → 10 publishes equal to hold), `test_ramp_reaches_waypoint_after_dt` (with `RateLimiter` limits generous), `test_leash_freezes_ramp` (measured stuck 0.6 rad behind → target stops advancing, `leash_engaged` true, `leash_events` ≥ 1), `test_limit_clamp_applied_before_adapter` (target beyond `q_max` with sign −1 → published = adapter(clamped)), `test_relative_adapter_identity_until_activate`, `test_step_exception_sets_fault_and_keeps_publishing` (smoother raises on cycle 5 → thread alive, last command republished, `healthy()` false), `test_thread_publishes_at_rate` (real thread 1.0 s → 85–115 publishes per side, `max_gap_s` < 0.05; tolerant because CPU may be busy with training).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add command adapter and 100 Hz joint streamer`.

### Task 12: Takeover, release and prepare logic

**Files:**
- Create: `robot_client/takeover.py`, `robot_client/prepare.py`, `tests/test_takeover.py`, `tests/test_prepare.py`

**Interfaces:**
- Protocols: `ControllerManager` — `list_controllers() -> dict[str, str]` (name → state), `switch(activate: list[str], deactivate: list[str], timeout_s: float) -> bool`. `GraphView` — `count_publishers(topic: str) -> int`, `measured(side) -> np.ndarray | None`. `MotionBackend` — `ptp(side, joints, max_vel, tol, timeout_s) -> bool`, `spine(height, vel, acc, dec, timeout_s) -> bool`, `gripper(side, width: float) -> None`.
- `class TakeoverError(RuntimeError)`.
- `GraphView` additionally: `foreign_messages(topic: str, window_s: float) -> int` — messages seen on the command topic whose `header.frame_id != "robot_client"` (all our command messages carry that frame id).
- `takeover(side, arm: ArmConfig, cm, graph, streamer, adapter, tk: TakeoverConfig, *, confirm_s=1.0, sleep=time.sleep) -> bool` (returns `activated_by_us`) — order:
  1. measured stream present for this side (else error);
  2. at least `tk.discovery_wait_s` since node start (DDS discovery);
  3. foreign **messages** on `command_topic` during a 1 s window == 0 (else error naming the topic, "is GELLO teleop still running?"); an idle foreign publisher endpoint only warns;
  4. the streamer is running with this side's hold seeded from measured (`Session` does that); wait `cycles_since_start(side) ≥ tk.warm_cycles`; `|last_command − measured|max ≤ tk.tol_rad`;
  5. by controller state from `list_controllers` (all calls with `tk.service_timeout_s`): `inactive` → `adapter.activate(...)`, `cm.switch([controller], [])`, confirm `active` within `confirm_s` and still within tol, return True; already `active` → warn, `adapter.activate(...)`, return False (we never deactivate what we didn't activate); missing/`unconfigured` → `TakeoverError` telling the operator to load/configure it with their tooling or set `takeover.mode: external`. In `external` mode: never switch; print "activate <controller> now", wait (up to 60 s, Ctrl-C aborts) for `active`, then `adapter.activate(...)`, return False.
- `release(side, arm, cm, streamer, *, activated_by_us: bool, hold_s=0.5, sleep=time.sleep) -> list[str]` — `streamer.hold(side)`, sleep `hold_s`, deactivate only if `activated_by_us`; collects errors instead of raising; returns error strings. Publishing keeps going until `Session` stops the streamer.
- `class Session(cfg, sides, cm, graph, streamer, adapters)` context manager. `sides` = `active_arms`, plus the inactive arm when `takeover.hold_inactive_arms` (it is taken over and only ever streams hold = its measured pose; spec "non-active arm is held at its measured pose"). `__enter__` starts the streamer with hold = measured for all `sides` (refuses if any measured is missing), then `takeover` per side, checking `streamer.healthy()` throughout; `__exit__` runs `release` for exactly the sides it activated (every side attempted even if one errors), then `streamer.stop()`, re-raises the original exception, logs release errors to stderr and the recorder.
- `MotionBackend` additionally: `spine_position(timeout_s) -> float | None`.
- `prepare(cfg: SiteConfig, cm, backend, streamer_active: bool) -> PrepareReport` — refuses if `streamer_active`; deactivates each arm's impedance controller if active (only in `takeover.mode: switch`; in `external` mode asks the operator to deactivate and waits for `inactive`); PTP left then right to `start_pose` with `prepare.ptp_*`; spine: reads `spine_position`, refuses if unavailable or `|height − current| > spine.max_delta_m`, else moves if `spine.enabled`; grippers 1.0 republished at 10 Hz for 1 s (a single message right after publisher creation can be lost). All calls have timeouts. Report lists each step's result; any failure stops and reports which step. `prepare` never runs without an operator Enter (CLI, Task 17).

- [ ] **Step 1: Write failing tests** — with `FakeCM` (records calls in order) and `FakeGraph`: `test_takeover_order` (switch is called only after ≥ 20 hold publishes equal to measured, and `adapter.activate` right before switch), `test_refuses_foreign_messages_warns_on_idle_endpoint`, `test_already_active_not_switched_not_deactivated`, `test_unconfigured_controller_points_to_external_mode`, `test_external_mode_never_switches`, `test_service_timeout_raises_takeover_error`, `test_spine_refused_on_large_delta_or_no_position`, `test_inactive_arm_held_when_configured`, `test_refuses_missing_controller_lists_available`, `test_activation_not_confirmed_raises_and_releases`, `test_interrupt_during_takeover_releases_only_what_we_activated` (KeyboardInterrupt raised from the fake sleep after left activated, before right: `__exit__` deactivates only left, original KeyboardInterrupt propagates), `test_release_attempts_all_sides_on_error`, `test_prepare_sequence_and_stop_on_failure` (PTP right fails → spine not called, report names `ptp right`), `test_prepare_refuses_while_streaming`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add controller takeover/release and prepare sequencing`.

### Task 13: Diagnostics recorder

**Files:**
- Create: `robot_client/recorder.py`, `tests/test_recorder.py`

**Interfaces:**
- `RunRecorder(out_dir: Path, run_name: str, config: dict, server_metadata: dict | None)` — creates `<out_dir>/<run_name>/` with `config.yaml`, `server_metadata.json`; `event(kind: str, **fields)` appends JSON lines to `events.log` (with wall + mono time); `episode(idx: int) -> EpisodeRecorder`.
- `EpisodeRecorder`: `step(record: dict)` → `steps.jsonl`; `chunk(obs_step, t_request, t_receive, latency_steps, chunk)` buffered, saved to `chunks.npz` on finish; `images(step, images: dict[str, np.ndarray])` saves JPEG (quality 90) only when `step % image_every == 0` (`image_every = control_hz // 2`); `finish(outcome: Literal["s", "f", "x", "freeze"], note: str, stats: dict, stream_stats: dict)` writes `result.json` and `stream_stats.json`. Writes survive a crash mid-episode (line-buffered jsonl; `finish` is idempotent).

- [ ] **Step 1: Write failing tests** — `test_layout_of_run_dir` (tmp_path; files exist, api key redacted in config.yaml), `test_images_at_2hz` (40 steps → 4 jpegs per camera), `test_finish_idempotent_and_chunks_npz_shapes`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add robot client diagnostics recorder`.

### Task 14: Episode runner

**Files:**
- Create: `robot_client/runner.py`, `robot_client/keys.py`, `tests/test_runner.py`

**Interfaces:**
- `keys.KeySource` protocol `poll() -> str | None`; `TerminalKeys(stream=sys.stdin)` (cbreak via termios when a TTY, restores on close; line mode otherwise); `FakeKeys(script: dict[int, str])` for tests (key delivered at a given tick).
- `Runner(cfg: SiteConfig, node: ObsSource, worker: InferenceWorker, streamer: Streamer, grippers: Callable[[str, float], None], recorder: EpisodeRecorder, keys: KeySource, clock=time.monotonic, sleep=time.sleep)`; `ObsSource` protocol: `snapshot() -> Snapshot`.
- `run_episode(generation: int) -> EpisodeOutcome(ended_by: Literal["timeout", "operator", "freeze"], reason: str, steps: int, stats: dict)`.
- Loop on a 20 Hz deadline grid; if a tick finishes more than one period late, re-anchor (`next = now + period`) and count `overruns` (no catch-up ticks).
- Per tick: key (`p` → paused/hold, Enter resumes, `q` → end) → `streamer.healthy()` (false → FREEZE with the fault text) → `safety.evaluate(snapshot.ages, dry_s, worker.error is not None, cfg.safety)` → mode. Ages of not-required streams are excluded from evaluation.
  - Leash: while `streamer.leash_engaged(side)` for an active arm, the runner does not pop new actions (the step counter still advances for async latency accounting; counted as `leash_steps`).
  - RUN, sync: `ChunkExecutor(execute_horizon, action_dim=16)`; when empty and nothing in flight, `worker.submit(generation_epoch, step, build_observation(...))`; streamer holds while waiting (counted as `wait_steps`, not HOLD).
  - RUN, async: `AsyncChunkBuffer(..., action_dim=16, arm_dims=ARM_DIMS, blend_dims=ARM_DIMS)`.
  - Chunks pass `layout.validate_chunk`. The first chunk after start or after any HOLD/pause is checked with `start_offset` against measured for active arms; > `max_start_offset_rad` → end the episode as `freeze` with reason.
  - Each executed action: `clamp_step` vs previous waypoint per active arm (`max_step_rad`), `streamer.set_waypoint(side, q)`; non-active arms never receive waypoints; gripper published for active arms when |Δ| > `gripper_deadband`, clipped to [0, 1].
  - HOLD: `streamer.hold()`, drop the queue, bump the generation epoch (in-flight results discarded), resume and replan when RUN returns.
  - FREEZE: return `ended_by="freeze"`.
  - Timeout at `episode_timeout_s`.
  - Each tick → `recorder.step({...})`; each chunk → `recorder.chunk(...)`; images via `recorder.images`.

- [ ] **Step 1: Write failing tests** (fake node, fake worker that returns chunks after N ticks, fake clock advancing per `sleep`, real `Streamer.step` not started — use a `FakeStreamer` recording calls):
  - `test_sync_executes_horizon_then_requests` (30 waypoints per chunk, one in-flight request at a time).
  - `test_async_blend_favor_old_used`.
  - `test_inactive_arm_gets_no_waypoints` (`active_arms=("right",)`).
  - `test_stale_camera_holds_then_replans` (age 1.2 s for 3 ticks → hold called, queue cleared, new request after recovery with a new epoch; an old-epoch result arriving later is ignored).
  - `test_server_loss_freezes_then_release` (worker returns errors; after 5 s simulated → `ended_by="freeze"`).
  - `test_start_offset_refused`.
  - `test_overrun_reanchors_without_burst` (one tick takes 0.3 s → next tick at now+0.05, `overruns == 1`, exactly one waypoint emitted for that tick).
  - `test_pause_and_quit_keys`.
  - `test_step_clamp_and_gripper_deadband` (plus a 1 Hz gripper keepalive republish).
  - `test_timeout_ends_episode`.
  - `test_leash_pauses_consumption` (fake streamer reports leash for 10 ticks → no waypoints emitted, then resumes from the next unconsumed action, not a skipped-ahead one).
  - `test_unhealthy_streamer_freezes`.
  - `test_pinned_state_sent_to_server` (worker receives state[7] == 0.0).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add robot client episode runner with sync and async modes`.

### Task 15: Policy link with timeouts and stub policy server

**Why:** `openpi_client.WebsocketClientPolicy` waits forever on `ConnectionRefusedError`, has no receive timeout, and other connect errors escape `PolicyClient`'s reconnect and kill the `InferenceWorker` for good.

**Files:**
- Create: `robot_client/policy_link.py`, `robot_client/stub_server.py`, `tests/test_policy_link.py`, `tests/test_stub_server.py`

**Interfaces (`policy_link.py`):**
- `RobotPolicyConnection(host, port, api_key, prompt, connect_timeout_s, infer_timeout_s)` — openpi protocol over `websockets.sync.client.connect(uri, open_timeout=connect_timeout_s, additional_headers=api-key header as openpi_client does, compression=None, max_size=None)`; reads metadata on connect and runs `layout.validate_server_metadata(meta, prompt)` (so **every reconnect is re-validated**); `get_server_metadata()`, `infer(obs)` with `recv(timeout=infer_timeout_s)`; a str reply → `RuntimeError(server traceback)`. Connection failures raise `PolicyUnreachableError("cannot reach policy server at host:port within Ns: <reason>")`; `LayoutMismatchError` propagates.
- `make_policy_client(cfg: SiteConfig) -> PolicyClient` — `eval_client.inference.PolicyClient(..., client_factory=lambda: RobotPolicyConnection(...))`.
- `ensure_worker(worker: InferenceWorker | None, cfg) -> InferenceWorker` — returns the worker if alive, else builds a fresh client + worker (called before every episode, so "next episode reconnects" holds); `LayoutMismatchError` is fatal for the run (exit 2).

**Interfaces:**
- `python -m robot_client.stub_server --port 8000 --layout munich16|sim20 --latency-s 0.5 --amplitude 0.05`: websocket server speaking the openpi protocol (send `msgpack_numpy` metadata on connect; per request unpack obs, reply `{"actions": (50, W), "server_timing": {"infer_ms": ...}}`; on exception send the traceback string). Metadata: `layout.POLICY_METADATA | {"config": "stub", "checkpoint": "stub"}` for munich16; `{"config": "stub-sim20"}` for sim20 (width 20).
- Chunk for munich16: every row = the request's `observation/state`, plus `amplitude * sin(2π t / 2.5 s)` on right joint 7 (index 14), grippers 1.0.

- Stub flags also: `--hang-after N` (stop replying after N requests, for timeout tests).

- [ ] **Step 1: Write failing tests** — start the stub in a thread on a free port; `make_policy_client` gets validated metadata; one `infer` returns `(50, 16)` with row 0 within `amplitude` of the sent state; the sim20 stub raises `LayoutMismatchError` at connect; nothing listening → `PolicyUnreachableError` within `connect_timeout_s` + 1 s; `--hang-after 1` → second infer fails within `infer_timeout_s` + 1 s and the worker stays alive (error result); stub restarted as sim20 between requests → reconnect raises `LayoutMismatchError`; `ensure_worker` replaces a dead worker.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add timeout-bounded policy link and stub policy server`.

### Task 16: ROS adapters

**Files:**
- Create: `robot_client/ros_io.py`, `tests/test_ros_io.py` (all `@pytest.mark.ros`)

**Interfaces:**
- `RobotNode(cfg: SiteConfig, *, command_mode: Literal["live", "preview", "none"])` (rclpy `Node`, spun by a `MultiThreadedExecutor` in a daemon thread via `start()`/`shutdown()`):
  - Subscriptions with `qos_profile_sensor_data` (best-effort, works with both reliable and best-effort publishers; Hajimi's Munich client does the same) for images **and** all JointState streams; a reliable/depth-10 subscriber on the command topic only for `foreign_messages`; callbacks decode and store latest sample + `time.monotonic()`; per-stream counters (count, max gap, last raw shape/encoding, decode errors).
  - `snapshot() -> Snapshot` (implements `ObsSource`); `stream_stats() -> dict`; `count_publishers(topic) -> int` (implements `GraphView`); `measured(side)`.
  - `publish_arm(side, q)` — `live`: JointState on `command_topic` with `command_joint_names or joint_names`, header stamp now, `header.frame_id = "robot_client"`; `preview`: on `/<side>/robot_client_preview/joint_states`; `none`: no publisher created (checks).
  - `publish_gripper(side, width)` — Float32; suppressed in `preview`/`none`.
- `RosControllerManager(node, namespace, timeout_s)` implements `ControllerManager` with `controller_manager_msgs/srv/ListControllers` and `SwitchController` (`activate_controllers`, `deactivate_controllers`, `strictness=STRICT`, `activate_asap=True`, `timeout`). Every call is `call_async` + bounded wait (never `call()`); timeout → `TakeoverError` naming the service.
- `RosMotionBackend(node, cfg)` implements `MotionBackend` with `franka_msgs/action/PTPMotion` (`goal_joint_configuration`, `maximum_joint_velocities`, `goal_tolerance`), `franka_spine_msgs/action/MoveAbsolute` (`position`, `velocity`, `acceleration`, `deceleration`) and `franka_spine_msgs/srv/GetPosition`; goal acceptance and result each bounded by `prepare.timeout_s`; cancel the goal on timeout or KeyboardInterrupt. Field names verified in Task 18 Step 2.

- [ ] **Step 1: Write failing ROS tests** (run in the container, Task 18 builds it; until then run them in Task 18): in-process publisher publishes a bgra8 Image and a reordered JointState → `snapshot()` has RGB image and mapped joints, ages < 0.2 s; `count_publishers` counts an extra publisher; `preview` mode publishes on the preview topic and creates no command-topic publisher.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Commit `Add robot client ROS node and controller/motion adapters` (tests executed in Task 18).

### Task 17: Checks and CLI

**Files:**
- Create: `robot_client/checks.py`, `robot_client/cli.py`, `robot_client/__main__.py`, `tests/test_checks.py`, `tests/test_cli.py`

**Interfaces:**
- `CheckReport(name, passed: bool, lines: list[str], data: dict)`; printed as a plain table; saved to `<run_dir>/<check>.json`.
- `check_obs(node: ObsSource & stats, cfg, seconds=8.0, out_dir, norm_stats_path: str | None = None, backend: MotionBackend | None = None) -> CheckReport` — per stream rate (warn below 80 % of expected: cameras 20/30/30 Hz, joints ≥ 100 Hz), raw shape/encoding, preprocessed shape, joint-name match, gripper value, `state_range_report` against the training norm stats if given, current spine position vs `spine.height`; saves one preprocessed JPEG per camera. Fails on any missing required stream, missing joint, or unsupported encoding.
- `check_infer(cfg, n=3, snapshot: Snapshot | None) -> CheckReport` — via `make_policy_client` (timeouts, metadata validation), chunk shape, latency mean/max; warns when RTT > `execute_horizon / control_hz` (sync) or > `trigger_len / control_hz` (async); zeros images/state when no snapshot.
- `check_motion(side, joints: Sequence[int] | Literal["all"] = "all", delta=0.05, ...) -> CheckReport` — under a `Session` for one arm, for each joint: ramp +delta over 1 s, hold 1 s, measure Δ, ramp back; verdict per joint `follows` (ratio 0.5–1.5), `reversed` (ratio −1.5 to −0.5), `no-motion`/`unclear`; prints the `joint_signs` line to paste. Aborts (release) if any |measured − start| > 2·delta + 0.02.
- CLI: `python -m robot_client <subcommand> --site PATH (default $ROBOT_SITE or /config/robot_site.yaml) [--set k=v ...] [--run-name NAME]`. Subcommands: `check-obs [--seconds]`, `check-infer [--n]`, `check-motion --arm {left,right} [--joint all|1..7] [--delta]`, `prepare`, `run [--episodes N] [--dry-run] [--arms ...]`, `release [--arm ...]` (deactivates the configured controller even if not activated by us — explicit operator recovery tool). Exit codes: 0 ok, 2 check failed / refused, 3 episode ended in FREEZE, 4 config error. Signal handling: SIGINT/SIGTERM → `KeyboardInterrupt` path so `Session.__exit__` releases; **during release, further SIGINTs are ignored** with "releasing — wait, or use the E-stop".
- `run` startup (once): load config → `check_ros_env` → `ensure_worker` (metadata validated; refuse on mismatch before anything moves) → `RunRecorder`.
- `run` flow per episode (live):
  1. "Reset the scene. Workspace clear? [Enter] move to start pose / [k] keep current pose / [q] quit" → `prepare` (or skip on `k`).
  2. `ensure_worker` → `Session` (takeover).
  3. "Press Enter to start the policy (q to release and quit)" — while waiting, the streamer holds and `healthy()`/safety are polled; failure → release.
  4. `Runner.run_episode` → `Session` exit (release).
  5. Outcome prompt `[s]uccess / [f]ail / [x] skip` + note → recorder `finish` → back to 1.
- `run --dry-run`: **no `prepare`, no `Session`, no gripper commands, no command-topic publishers** (node in `preview` mode); streamer publishes to preview topics with hold = measured; otherwise steps 3–5 are identical.

- [ ] **Step 1: Write failing tests** — `check_infer` against the Task 15 stub (pass for munich16, fail for sim20 with the metadata message); `check_obs` with a fake node missing a joint (fail, names the joint) and with a missing optional left gripper (pass); `check_motion` verdict math on synthetic Δ; CLI parse: `--set` repeated, exit code 4 on a bad YAML; `test_dry_run_builds_no_motion_objects` (monkeypatched factories: node in `preview` mode, `prepare`/`Session`/gripper publisher never constructed); `test_run_refuses_wrong_server_before_prepare` (sim20 stub → exit 2, `prepare` never called); `test_prepare_waits_for_enter` (FakeKeys with no Enter → `prepare` not called); `test_second_sigint_does_not_abort_release`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `Add robot client checks and CLI`.

### Task 18: Docker image and compose

**Files:**
- Create: `examples/ebim_task2/Dockerfile.robot_client`, `examples/ebim_task2/compose.robot_client.yml`, `examples/ebim_task2/robot_client/entrypoint.sh`

**Interfaces:**
- Dockerfile (build context = openpi root): stage `msgs` from `ros:humble-ros-base` clones `frankarobotics/franka_ros2` at `b4164f5` and `colcon build --packages-select franka_msgs franka_spine_msgs` (plus their in-repo msg deps if colcon asks); final stage `ros:humble-ros-base` + `ros-humble-rmw-fastrtps-cpp ros-humble-controller-manager-msgs python3-opencv python3-pip`, copies the install space, `pip install ruckig pyyaml pytest ./packages/openpi-client`, copies `examples/ebim_task2/{eval_client,robot_client}` to `/app` (readable). `ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=0 PYTHONPATH=/app`. Ships `robot_client/fastdds_udp_only.xml` (UDP transport only, no SHM); `FASTRTPS_DEFAULT_PROFILES_FILE` is passed through from the environment (compose: `${FASTRTPS_DEFAULT_PROFILES_FILE:-}`; README tells the operator to set it to the shipped file if their hosts use the UDP-only profile, as the Munich hosts do). Dev services (fake-robot, stub, client in separate containers) set it to the shipped file so they discover each other. `ENTRYPOINT ["/app/robot_client/entrypoint.sh"]` (sources both setups, prints `check_ros_env` warnings, `exec python3 -m robot_client "$@"`; `pytest`/`bash` pass through).
- Compose service `robot-client`: `network_mode: host`, `stdin_open: true`, `tty: true`, env `POLICY_SERVER_HOST/PORT/API_KEY`, `ROBOT_SITE=/config/robot_site.yaml`, volumes `./robot_client/robot_site.munich.yaml:/config/robot_site.yaml:ro` (operator swaps the file), `${OUT_DIR:-./robot_runs}:/out`. Service `fake-robot` (same image, `python3 -m robot_client.fake_robot`, profile `dev`) and `stub-server` (profile `dev`).

- [ ] **Step 1:** `docker compose -f examples/ebim_task2/compose.robot_client.yml build` — succeeds on x86_64.
- [ ] **Step 2:** `docker compose ... run --rm robot-client bash -lc "ros2 interface show franka_msgs/action/PTPMotion; ros2 interface show franka_spine_msgs/action/MoveAbsolute; python3 -c 'import numpy, rclpy, ruckig, cv2, openpi_client; print(numpy.__version__)'"` — the goal fields used in Task 16 are present (fix Task 16 if not); imports succeed.
- [ ] **Step 3:** Two separate invocations (the two suites share test module basenames and `conftest.py` names, so one rootless run hits "import file mismatch"): `docker compose ... run --rm robot-client pytest /app/robot_client/tests` and `... pytest /app/eval_client/tests` — all PASS including `ros` tests from Task 16 and the ruckig tests. Use the same split for the uv runs.
- [ ] **Step 4:** Commit `Add robot client Docker image and compose file`.

### Task 19: Fake robot and ROS integration tests

**Files:**
- Create: `robot_client/fake_robot.py`, `robot_client/tests/test_integration_ros.py` (`ros` marker)

**Interfaces:**
- `python -m robot_client.fake_robot --site PATH [--episode-dir <munich raw dataset> --episode 0] [--events /out/fake_robot_events.jsonl]`: publishes the three cameras (raw `bgra8` head at 1280×720, `rgb8` wrists 640×480; frames from the episode videos if given, else synthetic gradients) at 20/30/30 Hz; measured joints (`fr3v2` names, 200 Hz) that track the command topic with a first-order lag (`--tau`, default 0.05 s; `--tau 0.4` = slow controller that exercises the leash) **only while the fake `joint_impedance_controller` is active**; `--command-mode relative --signs 1,-1,1,1,1,1,1` makes the fake controller interpret commands as `q_act + signs ⊙ (msg − msg_at_activation)`; `--left-gripper-state off` omits the left gripper state topic; `--controller-state unconfigured|active` sets the initial controller state; gripper knuckle state following width commands (`0.8·(1−w)`); `list_controllers`/`switch_controller` services per side; PTP and spine action servers that ramp to the goal. Every command, switch and action is appended to the events file with mono time.
- Integration tests (each starts `fake_robot` and the stub server as subprocesses on `ROS_DOMAIN_ID=42`):
  - `test_takeover_ordering_and_continuous_stream`: `run --episodes 1` with `FakeKeys`/stdin script, `episode_timeout_s=5`: from the events file, the first `switch activate` comes after ≥ 20 commands equal to measured (≤ 0.02 rad), the largest command gap while active < 0.05 s, `deactivate` precedes the last command, and no command after `stop`.
  - `test_dry_run_never_moves_anything`: no switch, PTP, spine or gripper events and no command-topic messages; the preview topic receives messages.
  - `test_relative_mode_end_to_end`: fake robot in relative mode with a −1 sign and client `command_mode: relative_to_activation` with matching `joint_signs` → measured follows the policy target (stub sinusoid) within 0.02 rad; with mismatched signs, `check-motion` reports `reversed` for that joint.
  - `test_slow_controller_engages_leash_without_jump`: `--tau 0.4` → `leash_events` > 0 and no command step > `max_step_rad` after the leash releases.
  - `test_missing_left_gripper_state_runs`: `--left-gripper-state off` → `check-obs` passes, an episode runs.
  - `test_refuses_when_foreign_publisher`: a second publisher on `/right/gello/joint_states` → exit code 2, message names the topic.
  - `test_checkpoint_swap_same_image`: run `check-infer` against stub munich16 (pass) then restart the stub as sim20 (fail, exit 2) without changing the client config.
  - `test_server_killed_mid_episode_releases`: kill the stub after 2 s → episode ends `freeze`, events show deactivate.

- [ ] **Step 1:** Write the tests (they fail: no fake robot).
- [ ] **Step 2:** Implement `fake_robot.py`.
- [ ] **Step 3:** `docker compose ... run --rm robot-client pytest /app/robot_client/tests -m ros -v` — PASS.
- [ ] **Step 4:** Commit `Add fake robot and ROS integration tests for the robot client`.

### Task 20: Operator and server documentation

**Files:**
- Create: `examples/ebim_task2/README_ROBOT_CLIENT.md`, `examples/ebim_task2/SERVER.md`

**Content (fixed sections):**
- README: what this is (diagnostic run, one shot); assumptions (robot brought up and enabled by the organizers, arms/spine/grippers/cameras running, GELLO teleop stopped, operator at E-stop); build/run commands; command order `check-obs → check-infer → check-motion --arm right → check-motion --arm left → prepare → run --dry-run → run`; keys (Enter, `p`, `q`, Ctrl-C); a table **message → meaning → knob/action** generated from every `ConfigError`, `TakeoverError`, check failure and FREEZE reason string in the code (grep them; the table must cover all); the unconfirmed site facts (spec §9) each with the check that answers it and the knob (incl. `takeover.mode: external`, `state.pin`, camera `crop`, `FASTRTPS_DEFAULT_PROFILES_FILE`); recovery: if the container was killed (SIGKILL, power) while a controller was active, run `release --arm left --arm right`; how to send us the run directory.
- SERVER.md: build `scripts/docker/serve_policy.Dockerfile`, fetch checkpoint (`huggingface-cli download Ching-Yeh/ebim-task2 --include "<config>/<exp>/<step>/*"`), run with `POLICY_CONFIG` and `/ckpt` mount, expected log lines (warmup done), swap procedure, health check with `check-infer`, GPU memory expectation (~8–10 GB).

- [ ] **Step 1:** Write both; run a grep that lists every raised/printed operator-facing message and check each appears in the README table.
- [ ] **Step 2:** Commit `Add operator README and policy server notes for the robot client`.

### Task 21: Real-checkpoint integration, parity and release checklist

(After the stand-in run finishes; the 3090 can't train and serve at once.)

- [ ] **Step 1:** Hold-out eval on every saved step (Task 5 script); pick the step per `TRAINING_REAL.md` §5; upload it with `upload_checkpoint.sh`.
- [ ] **Step 2:** Serve it with `serve_munich.sh` from a fresh clone (local GPU), once normally and once with the tokenizer cache pre-filled and no container network, both reaching "Warmup done"; then from the client image: `check-infer` PASS (metadata `layout=munich16`, chunk (50,16), latency recorded) and `check-obs --norm-stats <ckpt>/assets/.../norm_stats.json` against the fake robot shows no out-of-range dims.
- [ ] **Step 3:** Parity: `fake_robot --episode-dir ... --episode <holdout id>` + `run --dry-run --episodes 1`; compare the first chunk the client received to `eval_real_holdout`'s prediction on the same frame — arm max abs diff < 0.02 rad (JPEG/decode differences only).
- [ ] **Step 4:** Swap test with real checkpoints: restart the server on a different step; same client container passes `check-infer` and records the new `checkpoint` id in `server_metadata.json`.
- [ ] **Step 5:** Release checklist (spec §7.1): every item ticked in the commit message; clean clone of the branch HEAD builds the image (`git clone --branch ebim/task2-real-robot-client <local repo> /tmp/...` then compose build). Commit `Release EBIM task2 real-robot client v1 (stand-in checkpoint <step>)`. Do not push.
