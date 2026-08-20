"""Episode orchestration: guards, sim-time pacing, sync/async rollout loops,
and the scene-reset handshake.

Operates on a node object (eval_client.ros_node.EvalClientNode in
production, fakes in tests) through a small duck-typed interface:
sim_now, reset_count, streams_fresh, missing_streams, snapshot,
publish_action, publish_stop, request_reset,
reset_request_has_subscriber, last_reset_event_wall,
clock_restarted_after, streams_fresh_since.
"""

import logging
import time

import numpy as np

from eval_client import obs as obs_mod
from eval_client.constants import CONTROL_HZ
from eval_client.inference import InferenceFailedError

# Canonical abort reasons (substrings of the raised messages).
EXTERNAL_RESET = "external scene reset"
CLOCK_STALL = "sim clock stalled"
CLOCK_BACKWARD = "sim clock jumped backward"
STALE_STREAMS = "stale observation streams"
QUEUE_DRY = "action queue dry"
NO_CLOCK = "no sim clock"
INCOMPLETE_OBS = "incomplete observation snapshot"
FIRST_INFER_TIMEOUT_S = 30.0  # first call may hit server JIT warmup


class EpisodeAbortedError(RuntimeError):
    pass


def wait_for(predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def wait_sim_time(node, target: float, reset_baseline: int, timeout_s: float = 10.0) -> float:
    """Block until sim time reaches target; returns the current sim time.

    An unrequested scene reset arriving mid-wait aborts promptly (the
    scene rebuild stalls the clock, so a hotkey-5 reset usually surfaces
    here)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if node.reset_count() != reset_baseline:
            raise EpisodeAbortedError(EXTERNAL_RESET)
        sim_time = node.sim_now()
        if sim_time is not None and sim_time >= target:
            return sim_time
        time.sleep(0.002)
    raise EpisodeAbortedError(f"{CLOCK_STALL} (waited {timeout_s}s for t={target:.2f})")


def needs_reset_request(current_count: int, handled_count: int) -> bool:
    """True when every scene-reset event so far was handled by the client,
    i.e. no external reset (hotkey 5) arrived since -- the next episode
    must request its own. An unhandled event means the scene is already
    freshly reset."""
    return current_count <= handled_count


def check_guards(node, t0: float, reset_baseline: int, cfg):
    """Per-step guards shared by the sync and async loops. Returns
    (sim_time, done) where done means the episode timeout was reached;
    raises EpisodeAbortedError on any abnormal condition. Check order matters:
    a timed-out episode must finish cleanly even if streams just went
    stale."""
    if node.reset_count() != reset_baseline:
        raise EpisodeAbortedError(EXTERNAL_RESET)
    sim_time = node.sim_now()
    if sim_time is None:
        raise EpisodeAbortedError(NO_CLOCK)
    if sim_time - t0 >= cfg.episode_timeout:
        return sim_time, True
    if sim_time < t0 - 0.5:
        raise EpisodeAbortedError(f"{CLOCK_BACKWARD} (unexpected scene reset?)")
    if not node.streams_fresh():
        raise EpisodeAbortedError(f"{STALE_STREAMS}: {node.missing_streams()}")
    return sim_time, False


def snapshot_observation(node, cfg) -> dict:
    snapshot = node.snapshot()
    if snapshot is None:
        raise EpisodeAbortedError(INCOMPLETE_OBS)
    _, state, images = snapshot
    return obs_mod.make_observation(state, images, cfg.prompt)


def _base_record(steps: int, duration: float, num_infer_calls: int, infer_ms: list) -> dict:
    return {
        "steps": steps,
        "duration_s": round(duration, 3),
        "num_infer_calls": num_infer_calls,
        "infer_ms_mean": round(float(np.mean(infer_ms)), 1) if infer_ms else None,
        "infer_ms_max": round(float(np.max(infer_ms)), 1) if infer_ms else None,
    }


def run_episode_sync(node, policy_client, cfg) -> dict:
    """Blocking rollout: infer every replan_steps, freezing the control
    grid during inference (the schedule re-anchors after each call)."""
    from eval_client.chunking import ChunkExecutor

    executor = ChunkExecutor(cfg.replan_steps)
    step_period = 1.0 / CONTROL_HZ
    t0 = node.sim_now()
    if t0 is None:
        raise EpisodeAbortedError(NO_CLOCK)
    reset_baseline = node.reset_count()
    num_infer_calls = 0
    infer_ms = []
    steps = 0
    next_step_time = t0

    try:
        while True:
            _, done = check_guards(node, t0, reset_baseline, cfg)
            if done:
                break

            if executor.needs_infer:
                observation = snapshot_observation(node, cfg)
                start = time.monotonic()
                try:
                    result = policy_client.infer(observation)
                except InferenceFailedError:
                    # Connection was rebuilt; retry with a fresh snapshot on
                    # the next iteration (budget enforced by PolicyClient).
                    continue
                infer_ms.append((time.monotonic() - start) * 1000.0)
                num_infer_calls += 1
                executor.load(np.asarray(result["actions"]))
                # Re-anchor the 30 Hz schedule after the inference gap.
                next_step_time = node.sim_now()

            node.publish_action(executor.next_action(), cfg.pedal_threshold)
            steps += 1
            next_step_time += step_period
            wait_sim_time(node, next_step_time, reset_baseline)
    finally:
        node.publish_stop()

    return _base_record(steps, node.sim_now() - t0, num_infer_calls, infer_ms)


def run_episode_async(node, worker, cfg, generation: int) -> dict:
    """Async rollout: the worker thread owns the blocking websocket call;
    this loop keeps publishing queued actions on the 30 Hz sim grid and
    requests a fresh chunk before the queue runs dry. Unlike the sync
    loop it never re-anchors the schedule after the prime -- inference no
    longer stalls the grid."""
    from eval_client.chunking import AsyncChunkBuffer

    buffer = AsyncChunkBuffer(cfg.execute_horizon, cfg.chunk_size_threshold, cfg.chunk_merge)
    step_period = 1.0 / CONTROL_HZ
    t0 = node.sim_now()
    if t0 is None:
        raise EpisodeAbortedError(NO_CLOCK)
    reset_baseline = node.reset_count()
    num_infer_calls = 0
    infer_failures = 0
    infer_ms = []
    steps = 0

    # Responses left in flight by a prior aborted episode are stale.
    worker.purge()

    try:
        # Prime: block on the first chunk, mirroring the sync loop's first
        # inference.
        worker.submit(generation, 0, snapshot_observation(node, cfg))
        buffer.on_request_sent()
        num_infer_calls += 1
        deadline = time.monotonic() + FIRST_INFER_TIMEOUT_S
        while buffer.in_flight:
            if worker.error is not None:
                raise worker.error
            if node.reset_count() != reset_baseline:
                raise EpisodeAbortedError(EXTERNAL_RESET)
            if time.monotonic() > deadline:
                raise EpisodeAbortedError(f"first inference timed out ({FIRST_INFER_TIMEOUT_S:.0f}s)")
            for result in worker.drain(generation):
                if result.error is not None:
                    infer_failures += 1
                    buffer.in_flight = False
                    worker.submit(generation, 0, snapshot_observation(node, cfg))
                    buffer.on_request_sent()
                    num_infer_calls += 1
                else:
                    buffer.on_chunk(result.obs_step, result.actions, next_step=0)
                    infer_ms.append(result.infer_ms)
            time.sleep(0.005)
        # Single re-anchor after the priming gap.
        next_step_time = node.sim_now()

        while True:
            _, done = check_guards(node, t0, reset_baseline, cfg)
            if done:
                break
            if worker.error is not None:
                raise worker.error
            for result in worker.drain(generation):
                if result.error is not None:
                    # Connection was rebuilt inside PolicyClient; clearing
                    # in_flight lets should_request() re-fire with a fresh
                    # snapshot on the next step.
                    infer_failures += 1
                    buffer.in_flight = False
                else:
                    buffer.on_chunk(result.obs_step, result.actions, next_step=steps)
                    infer_ms.append(result.infer_ms)
            if buffer.should_request():
                worker.submit(generation, steps, snapshot_observation(node, cfg))
                buffer.on_request_sent()
                num_infer_calls += 1
            if buffer.consecutive_dry >= cfg.queue_dry_abort_steps:
                raise EpisodeAbortedError(f"{QUEUE_DRY} for {buffer.consecutive_dry} steps")
            action = buffer.pop()
            if action is not None:
                node.publish_action(action, cfg.pedal_threshold)
            steps += 1
            next_step_time += step_period
            wait_sim_time(node, next_step_time, reset_baseline)
    finally:
        node.publish_stop()

    record = _base_record(steps, node.sim_now() - t0, num_infer_calls, infer_ms)
    record.update(buffer.stats())
    record["async"] = True
    record["infer_failures"] = infer_failures
    return record


def ensure_fresh_scene(node, handled_count: int, timeout_s: float = 30.0) -> int:
    """Make sure the scene is freshly reset; returns the new handled reset
    count.

    If an external reset event (hotkey 5) arrived since handled_count was
    taken, the scene is already reset -- skip the request and only wait
    for the post-reset clock. Otherwise request a reset and wait for its
    completion event. Confirmation is keyed on the event's arrival wall
    time: the event is published after world.reset() completes, so a
    clock message after it is necessarily post-reset."""
    before = node.reset_count()
    if needs_reset_request(before, handled_count):
        # A request published before DDS discovery matches the sim-side
        # subscription is silently dropped; wait for the match first.
        if not wait_for(node.reset_request_has_subscriber, timeout_s=min(10.0, timeout_s)):
            raise RuntimeError("no subscriber on the scene-reset request topic (is the sim running with --record?)")
        node.request_reset()
        if not wait_for(lambda: node.reset_count() > before, timeout_s / 2):
            logging.warning("scene reset not confirmed yet; re-requesting")
            node.request_reset()
            if not wait_for(lambda: node.reset_count() > before, timeout_s / 2):
                raise RuntimeError(
                    f"scene reset not confirmed within {timeout_s}s (is the sim running with the reset controller?)"
                )
    else:
        logging.info("external scene reset detected; skipping reset request")
    # The sim clock restarts from zero on reset; wait for a clock message
    # from after the reset event so the episode t0 is not latched from the
    # old epoch.
    event_wall = node.last_reset_event_wall()
    if not wait_for(lambda: node.clock_restarted_after(event_wall), timeout_s):
        raise RuntimeError("sim clock did not restart after scene reset")
    return node.reset_count()
