import dataclasses
import io
import os
from pathlib import Path
import termios

from eval_client.inference import WorkerResult
import numpy as np
import pytest
from robot_client import layout
from robot_client.config import load_site_config
from robot_client.keys import ENTER
from robot_client.keys import FakeKeys
from robot_client.keys import TerminalKeys
from robot_client.preprocess import Snapshot
from robot_client.runner import Runner
from robot_client.runner import required_stream_keys

MUNICH_YAML = Path(__file__).resolve().parents[1] / "robot_site.munich.yaml"
PERIOD = 0.05
T0 = 1000.0
IMAGES = {name: np.full((24, 32, 3), 90, dtype=np.uint8) for name in ("head", "wrist_left", "wrist_right")}
KNUCKLE = {"left": 0.37, "right": 0.1}


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        assert dt >= 0
        self.t += dt


@dataclasses.dataclass
class Request:
    generation: int
    obs_step: int
    obs: dict
    tick: int
    pending_before: int  # results not yet delivered when this request was submitted


class FakeWorker:
    """Answers request n `delay_ticks` ticks after it was submitted; `respond(n) -> (delay_ticks, actions, error)`."""

    def __init__(self, rig, respond):
        self.rig = rig
        self.respond = respond
        self.requests: list[Request] = []
        self.pending: list[tuple[int, WorkerResult]] = []
        self.discarded: list[WorkerResult] = []
        self.error = None
        self.alive = True

    def submit(self, generation, obs_step, obs):
        n = len(self.requests)
        self.requests.append(Request(generation, obs_step, obs, self.rig.tick, len(self.pending)))
        delay, actions, error = self.respond(n)
        self.pending.append((self.rig.tick + delay, WorkerResult(generation, obs_step, actions, 12.0, error)))

    def drain(self, generation):
        ready = [r for t, r in self.pending if self.rig.tick >= t]
        self.pending = [(t, r) for t, r in self.pending if self.rig.tick < t]
        out = []
        for r in ready:
            (out if r.generation == generation else self.discarded).append(r)
        return out

    def is_alive(self):
        return self.alive


class FakeStreamer:
    def __init__(self, rig):
        self.rig = rig
        self.waypoints: list[tuple[int, str, np.ndarray]] = []
        self.holds: list[tuple[int, str | None]] = []
        self.fault = None
        self.healthy_fn = lambda tick: True
        self.leash_fn = lambda tick, side: False

    def set_waypoint(self, side, q):
        self.waypoints.append((self.rig.tick, side, np.array(q, dtype=float)))

    def hold(self, side=None):
        self.holds.append((self.rig.tick, side))

    def healthy(self):
        return self.healthy_fn(self.rig.tick)

    def leash_engaged(self, side):
        return self.leash_fn(self.rig.tick, side)

    def wps(self, side):
        return [(t, q) for t, s, q in self.waypoints if s == side]


class FakeRecorder:
    def __init__(self):
        self.steps, self.chunks, self.frames = [], [], []

    def step(self, record):
        self.steps.append(record)

    def chunk(self, obs_step, t_request, t_receive, latency_steps, chunk):
        self.chunks.append((obs_step, t_request, t_receive, latency_steps, np.asarray(chunk)))

    def images(self, step, images):
        self.frames.append((step, images))


class TickKeys(FakeKeys):
    """FakeKeys that drives the rig's tick counter (polled first thing in every tick)."""

    def __init__(self, rig, script):
        super().__init__(script)
        self.rig = rig

    def poll(self):
        rig = self.rig
        rig.tick += 1
        rig.tick_starts.append(rig.clock())
        rig.on_tick(rig.tick)
        return super().poll()


class FakeNode:
    def __init__(self, rig):
        self.rig = rig

    def snapshot(self):
        rig = self.rig
        ages = dict.fromkeys(("cam/head", "cam/wrist_left", "cam/wrist_right", "arm/left", "arm/right"), 0.01)
        ages["grip/right"] = 0.01
        ages["grip/left"] = float("inf")  # not required at Munich: must be ignored
        ages.update(rig.ages(rig.tick))
        ages = {k: v for k, v in ages.items() if v is not None}
        return Snapshot(
            images=dict(IMAGES),
            arm_q={side: q.copy() for side, q in rig.measured.items()},
            knuckle=dict(KNUCKLE),
            ages=ages,
            t_mono=rig.clock(),
        )


class Rig:
    def __init__(self, overrides=(), *, respond=None, keys=None, timeout=3.0):
        self.cfg = load_site_config(MUNICH_YAML, [f"safety.episode_timeout_s={timeout}", *overrides], {})
        self.measured = {side: np.array(self.cfg.arms[side].start_pose) for side in ("left", "right")}
        self.tick = -1
        self.tick_starts: list[float] = []
        self.clock = Clock()
        self.on_tick = lambda tick: None
        self.ages = lambda tick: {}
        self.worker = FakeWorker(self, respond or (lambda n: (1, self.chunk(), None)))
        self.streamer = FakeStreamer(self)
        self.recorder = FakeRecorder()
        self.grips: list[tuple[int, str, float]] = []
        self.keys = TickKeys(self, keys or {})
        self.runner = Runner(
            self.cfg,
            FakeNode(self),
            self.worker,
            self.streamer,
            lambda side, width: self.grips.append((self.tick, side, width)),
            self.recorder,
            self.keys,
            clock=self.clock,
            sleep=self.clock.sleep,
        )

    def chunk(self, tag=0.0, d=0.001, n=50, grip=(1.0, 0.5)):
        """Row i: both arms at measured + tag + i*d, gripper widths `grip`."""
        rows = np.zeros((n, layout.ACTION_DIM))
        ramp = tag + d * np.arange(n)[:, None]
        rows[:, layout.ARM_SLICES["left"]] = self.measured["left"] + ramp
        rows[:, layout.ARM_SLICES["right"]] = self.measured["right"] + ramp
        rows[:, layout.GRIPPER_INDEX["left"]] = grip[0]
        rows[:, layout.GRIPPER_INDEX["right"]] = grip[1]
        return rows

    def run(self, generation=1):
        return self.runner.run_episode(generation)


def row(chunk, i, side="right"):
    return chunk[i, layout.ARM_SLICES[side]]


# ----------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------


def test_sync_executes_horizon_then_requests():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk(tag=0.0, d=0.001))
        return 3, chunks[-1], None

    rig = Rig(respond=respond, timeout=4.025)  # ticks 0..80
    out = rig.run()

    assert out.ended_by == "timeout"
    assert out.steps == 81
    assert [r.tick for r in rig.worker.requests] == [0, 33, 66]
    assert [r.obs_step for r in rig.worker.requests] == [0, 33, 66]
    assert all(r.pending_before == 0 for r in rig.worker.requests)  # one in flight at a time
    for side in ("left", "right"):
        wps = rig.streamer.wps(side)
        ticks = [t for t, _ in wps]
        assert ticks == [*range(3, 33), *range(36, 66), *range(69, 81)]
        for k, start in enumerate((3, 36, 69)):
            for t, q in wps:
                if start <= t < start + 30:
                    np.testing.assert_allclose(q, row(chunks[k], t - start, side))
    assert out.stats["wait_steps"] == 3 * 3
    assert out.stats["hold_steps"] == 0
    assert out.stats["requests"] == 3


def test_async_blend_favor_old_used():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk(tag=0.05 * n, d=0.001))
        return 2, chunks[-1], None

    rig = Rig(["inference.mode=async"], respond=respond, timeout=1.5)
    out = rig.run()

    req_ticks = [r.tick for r in rig.worker.requests]
    assert req_ticks[:2] == [0, 17]  # first at once; next when 15 (= 0.5 * 30) actions remain
    wps = dict(rig.streamer.wps("right"))
    # First chunk after start: no latency skip (the robot stood still), rows 0.. from tick 2.
    for t in range(2, 19):
        np.testing.assert_allclose(wps[t], row(chunks[0], t - 2))
    # Second chunk (obs step 17, arrives at step 19 -> 2 stale rows skipped) blended 0.7 old / 0.3 new.
    a, b = chunks[0], chunks[1]
    np.testing.assert_allclose(wps[19], 0.7 * row(a, 17) + 0.3 * row(b, 2))
    np.testing.assert_allclose(wps[25], 0.7 * row(a, 23) + 0.3 * row(b, 8))
    assert out.stats["chunk_latency_steps_max"] == 2
    assert out.stats["switch_jump_max"] is not None


def test_inactive_arm_gets_no_waypoints():
    def respond(n):
        c = rig.chunk()
        c[:, layout.ARM_SLICES["left"]] += 1.0  # far away: must not trip the start check either
        return 1, c, None

    rig = Rig(["active_arms=[right]"], respond=respond, timeout=1.0)
    out = rig.run()

    assert out.ended_by == "timeout"
    assert rig.streamer.wps("left") == []
    assert len(rig.streamer.wps("right")) == out.steps - 1
    assert {side for _, side, _ in rig.grips} == {"right"}


def test_stale_camera_holds_then_replans():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk(tag=0.01 * (n + 1)))
        return (6 if n == 0 else 3), chunks[-1], None

    rig = Rig(respond=respond, timeout=1.0)
    rig.ages = lambda tick: {"cam/head": 1.2} if tick in (1, 2, 3) else {}
    out = rig.run()

    old, new = rig.worker.requests[:2]
    assert old.tick == 0
    assert rig.streamer.holds[0] == (1, None)
    assert new.tick == 4  # replanned once the camera recovered
    assert new.generation == old.generation + 1
    # The old request's answer (ready at tick 6) was discarded; nothing of it was executed.
    assert [r.obs_step for r in rig.worker.discarded] == [0]
    wps = rig.streamer.wps("right")
    assert wps[0][0] == 7
    for t, q in wps:
        np.testing.assert_allclose(q, row(chunks[1], t - 7))
    assert out.stats["hold_steps"] == 3
    assert out.stats["holds"] == 1


def test_hold_mid_chunk_drops_queue():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk(tag=0.01 * n))
        return 1, chunks[-1], None

    rig = Rig(respond=respond, timeout=1.0)
    rig.ages = lambda tick: {"arm/left": 1.2} if tick in (5, 6, 7) else {}
    rig.run()

    wps = dict(rig.streamer.wps("right"))
    assert sorted(wps)[:4] == [1, 2, 3, 4]
    assert not any(t in wps for t in (5, 6, 7, 8))
    assert [r.tick for r in rig.worker.requests] == [0, 8]
    np.testing.assert_allclose(wps[9], row(chunks[1], 0))  # not chunk 0's row 4


def test_server_loss_freezes_then_release():
    rig = Rig(respond=lambda n: (1, None, "connection reset"), timeout=60.0)
    out = rig.run()

    assert out.ended_by == "freeze"
    assert "inference dry" in out.reason
    assert 5.0 < rig.clock.t - T0 <= 5.1
    assert len(rig.worker.requests) > 50  # kept retrying with fresh observations
    assert rig.streamer.waypoints == []
    assert rig.streamer.holds[-1] == (out.steps, None)  # held on the way out; Session releases
    assert out.stats["infer_errors"] >= 50


def test_fatal_worker_error_freezes():
    rig = Rig(timeout=60.0)
    rig.on_tick = lambda tick: setattr(rig.worker, "error", RuntimeError("budget gone")) if tick == 10 else None
    out = rig.run()

    assert out.ended_by == "freeze"
    assert "fatal inference error" in out.reason
    assert "budget gone" in out.reason
    assert out.steps == 10


def test_start_offset_refused():
    def respond(n):
        c = rig.chunk()
        c[:, layout.ARM_SLICES["right"].start + 3] += 0.4
        return 2, c, None

    rig = Rig(respond=respond, timeout=10.0)
    out = rig.run()

    assert out.ended_by == "freeze"
    assert "right" in out.reason
    assert "max_start_offset_rad" in out.reason
    assert out.steps == 2
    assert rig.streamer.waypoints == []


def test_start_offset_checked_again_after_pause():
    def respond(n):
        c = rig.chunk()
        if n == 1:
            c[:, 0] += 0.4  # left joint 0
        return 1, c, None

    rig = Rig(respond=respond, keys={5: "p", 7: ENTER}, timeout=10.0)
    out = rig.run()

    assert out.ended_by == "freeze"
    assert "left" in out.reason
    assert out.steps == 8


def test_overrun_reanchors_without_burst():
    rig = Rig(timeout=1.5)

    def slow(tick):
        if tick == 10:
            rig.clock.t += 0.3

    rig.on_tick = slow
    out = rig.run()

    starts = rig.tick_starts
    assert starts[10] - starts[9] == pytest.approx(PERIOD)
    assert starts[11] - starts[10] == pytest.approx(0.3 + PERIOD)  # re-anchored: now + period
    assert starts[12] - starts[11] == pytest.approx(PERIOD)
    assert out.stats["overruns"] == 1
    per_tick = [t for t, _ in rig.streamer.wps("right")]
    assert per_tick == sorted(set(per_tick))  # never more than one waypoint per tick
    assert per_tick.count(10) == 1
    assert per_tick.count(11) == 1


def test_pause_and_quit_keys():
    rig = Rig(keys={5: "p", 10: ENTER, 40: "q"}, timeout=60.0)
    out = rig.run()

    assert out.ended_by == "operator"
    assert out.steps == 40
    assert rig.streamer.holds[0] == (5, None)
    assert rig.streamer.holds[-1][0] == 40  # held at the end
    wp_ticks = {t for t, _ in rig.streamer.wps("right")}
    assert not wp_ticks & set(range(5, 11))
    assert [r.tick for r in rig.worker.requests] == [0, 10]
    assert rig.worker.requests[1].generation == rig.worker.requests[0].generation + 1
    assert out.stats["pause_steps"] == 5


def test_quit_while_paused():
    rig = Rig(keys={3: "p", 6: "q"}, timeout=60.0)
    out = rig.run()
    assert (out.ended_by, out.steps) == ("operator", 6)


def test_step_clamp_and_gripper_deadband():
    def respond(n):
        c = rig.chunk(d=0.0, grip=(1.0, 0.5))
        j0 = layout.ARM_SLICES["right"].start
        c[1:, j0] += 0.25
        c[:, layout.GRIPPER_INDEX["right"]] = [0.5, 0.51, 0.53, 0.53, 1.3] + [1.3] * 45
        return 1, c, None

    rig = Rig(respond=respond, timeout=1.5)  # ticks 0..29: only the first chunk executes
    out = rig.run()

    j0 = rig.measured["right"][0]
    got = [q[0] - j0 for _, q in rig.streamer.wps("right")[:5]]
    np.testing.assert_allclose(got, [0.0, 0.1, 0.2, 0.25, 0.25], atol=1e-12)
    assert out.stats["step_clamps"] == 2
    right = [(t, w) for t, s, w in rig.grips if s == "right"]
    assert right[:3] == [(1, 0.5), (3, 0.53), (5, 1.0)]
    # 1 Hz keepalive republishes the last width.
    assert right[3][1] == 1.0
    assert right[3][0] - 5 in (20, 21)
    left = [(t, w) for t, s, w in rig.grips if s == "left"]
    assert left[0] == (1, 1.0)
    assert left[1][0] - 1 in (20, 21)


def test_timeout_ends_episode():
    rig = Rig(timeout=1.0)
    out = rig.run()
    assert out.ended_by == "timeout"
    assert 20 <= out.steps <= 21
    assert "episode_timeout_s" in out.reason
    assert rig.streamer.holds[-1][1] is None


def test_leash_pauses_consumption():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk())
        return 1, chunks[-1], None

    rig = Rig(respond=respond, timeout=1.5)
    rig.streamer.leash_fn = lambda tick, side: side == "right" and 5 <= tick < 15
    out = rig.run()

    wps = dict(rig.streamer.wps("right"))
    assert not any(t in wps for t in range(5, 15))
    for t in range(1, 5):
        np.testing.assert_allclose(wps[t], row(chunks[0], t - 1))
    np.testing.assert_allclose(wps[15], row(chunks[0], 4))  # the next unconsumed row, not row 14
    assert out.stats["leash_steps"] == 10
    assert len(rig.worker.requests) == 1


def test_async_leash_freezes_plan_time():
    chunks = []

    def respond(n):
        chunks.append(rig.chunk(tag=0.05 * n, d=0.001))
        return (2 if n == 0 else 6), chunks[-1], None

    rig = Rig(["inference.mode=async"], respond=respond, timeout=1.5)
    rig.streamer.leash_fn = lambda tick, side: side == "right" and 19 <= tick < 22
    out = rig.run()

    # Request at tick 17 (15 actions left) is answered at tick 23. Real progress since: ticks 17, 18, 22 -> skip 3,
    # not 6 (the three leash ticks do not count).
    assert rig.worker.requests[1].obs_step == 17
    assert rig.recorder.chunks[1][3] == 3
    wps = dict(rig.streamer.wps("right"))
    assert not any(t in wps for t in (19, 20, 21))
    a, b = chunks
    np.testing.assert_allclose(wps[22], row(a, 17))  # resumes with the next unconsumed row
    np.testing.assert_allclose(wps[23], 0.7 * row(a, 18) + 0.3 * row(b, 3))  # queue front aligned with fresh row 3
    assert out.stats["leash_steps"] == 3
    assert out.stats["stale_steps_skipped"] == 3


def test_unhealthy_streamer_freezes():
    rig = Rig(timeout=60.0)
    rig.streamer.healthy_fn = lambda tick: tick < 10
    rig.streamer.fault = "publish failed: boom"
    out = rig.run()
    assert out.ended_by == "freeze"
    assert "boom" in out.reason
    assert out.steps == 10


def test_pinned_state_sent_to_server():
    rig = Rig(timeout=0.5)
    out = rig.run()
    assert out.ended_by == "timeout"  # grip/left never seen, but it is not required
    obs = rig.worker.requests[0].obs
    state = obs["observation/state"]
    assert state.shape == (16,)
    assert state[7] == 0.0
    assert state[15] == pytest.approx(KNUCKLE["right"])
    np.testing.assert_allclose(state[8:15], rig.measured["right"], rtol=1e-6)
    assert obs["prompt"] == layout.PROMPT
    assert obs["observation/image"].shape == (224, 224, 3)


def test_required_stream_never_seen_freezes():
    rig = Rig(timeout=60.0)
    rig.ages = lambda tick: {"grip/right": None}  # absent from the snapshot
    out = rig.run()
    assert out.ended_by == "freeze"
    assert "grip/right never seen" in out.reason
    assert out.steps == 0


def test_required_stream_keys_follow_config():
    rig = Rig()
    assert required_stream_keys(rig.cfg) == (
        "cam/head",
        "cam/wrist_left",
        "cam/wrist_right",
        "arm/left",
        "arm/right",
        "grip/right",
    )


def test_wrong_width_chunk_freezes():
    rig = Rig(respond=lambda n: (1, np.zeros((50, 17)), None), timeout=60.0)
    out = rig.run()
    assert out.ended_by == "freeze"
    assert "width 17" in out.reason


def test_recorder_receives_steps_chunks_images():
    rig = Rig(timeout=1.0)
    out = rig.run()
    assert len(rig.recorder.steps) == out.steps
    assert rig.recorder.steps[0]["mode"] == "wait"
    assert rig.recorder.steps[1]["mode"] == "run"
    assert len(rig.recorder.steps[1]["action"]) == 16
    obs_step, t_req, t_rcv, latency, chunk = rig.recorder.chunks[0]
    assert (obs_step, latency, chunk.shape) == (0, 1, (50, 16))
    assert t_rcv - t_req == pytest.approx(PERIOD)
    assert [s for s, _ in rig.recorder.frames] == [0, 10]
    assert rig.recorder.frames[0][1]["head"].shape == (224, 224, 3)


def test_stale_epoch_from_previous_episode_ignored():
    rig = Rig(timeout=0.3)
    rig.run(generation=1)
    stale = WorkerResult(rig.worker.requests[-1].generation, 0, rig.chunk(tag=0.2), 1.0)
    rig.worker.pending.append((0, stale))
    rig.streamer.waypoints.clear()
    rig.run(generation=2)
    assert stale in rig.worker.discarded
    assert all(q[0] - rig.measured["right"][0] < 0.1 for _, q in rig.streamer.wps("right"))


# ----------------------------------------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------------------------------------


def test_fake_keys_deliver_at_tick():
    keys = FakeKeys({1: "p", 3: ENTER})
    assert [keys.poll() for _ in range(5)] == [None, "p", None, ENTER, None]


def test_terminal_keys_line_mode_stringio():
    keys = TerminalKeys(io.StringIO("p\n\nQ\nfoo\n"))
    assert [keys.poll() for _ in range(5)] == ["p", ENTER, "q", "f", None]
    keys.close()


def test_terminal_keys_line_mode_pipe():
    r, w = os.pipe()
    with os.fdopen(r, "r") as stream:
        keys = TerminalKeys(stream)
        assert keys.poll() is None
        os.write(w, b"p\nq")
        assert keys.poll() == "p"
        assert keys.poll() is None  # "q" without its newline is not a line yet
        os.write(w, b"\n\n")
        assert keys.poll() == "q"
        assert keys.poll() == ENTER
        os.close(w)
        assert keys.poll() is None  # EOF
        keys.close()


def test_terminal_keys_cbreak_restores_terminal():
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        with os.fdopen(os.dup(slave), "r") as stream:
            with TerminalKeys(stream) as keys:
                assert not termios.tcgetattr(slave)[3] & termios.ICANON
                assert keys.poll() is None
                os.write(master, b"p")
                assert keys.poll() == "p"
                os.write(master, b"\r")
                assert keys.poll() == ENTER
            assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)
