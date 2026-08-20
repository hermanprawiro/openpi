import threading
import time

from eval_client import episode
from eval_client import inference
from eval_client.config import EvalConfig
import numpy as np
import pytest


def make_config(**overrides) -> EvalConfig:
    defaults = {"num_episodes": 1, "episode_timeout": 0.4, "replan_steps": 5, "max_reconnects": 2}
    defaults.update(overrides)
    return EvalConfig(**defaults)


class FakeClockNode:
    """Node stub with a manually controlled clock and guard inputs."""

    def __init__(self, sim_time=0.0):
        self.sim_time = sim_time
        self.resets = 0
        self.fresh = True

    def sim_now(self):
        return self.sim_time

    def reset_count(self):
        return self.resets

    def streams_fresh(self, max_age_s=None):
        return self.fresh

    def missing_streams(self):
        return [] if self.fresh else ["image_head"]


class TestCheckGuards:
    def test_normal_step(self):
        node = FakeClockNode(sim_time=1.0)
        sim_time, done = episode.check_guards(node, t0=0.0, reset_baseline=0, cfg=make_config(episode_timeout=90.0))
        assert sim_time == 1.0
        assert done is False

    def test_timeout_reached(self):
        node = FakeClockNode(sim_time=90.5)
        _, done = episode.check_guards(node, 0.0, 0, make_config(episode_timeout=90.0))
        assert done is True

    def test_external_reset_beats_everything(self):
        node = FakeClockNode(sim_time=95.0)
        node.resets = 1
        node.fresh = False
        with pytest.raises(episode.EpisodeAbortedError, match=episode.EXTERNAL_RESET):
            episode.check_guards(node, 0.0, 0, make_config(episode_timeout=90.0))

    def test_timeout_beats_stale_streams(self):
        node = FakeClockNode(sim_time=95.0)
        node.fresh = False
        _, done = episode.check_guards(node, 0.0, 0, make_config(episode_timeout=90.0))
        assert done is True

    def test_backward_clock_aborts(self):
        node = FakeClockNode(sim_time=1.0)
        with pytest.raises(episode.EpisodeAbortedError, match="backward"):
            episode.check_guards(node, t0=5.0, reset_baseline=0, cfg=make_config(episode_timeout=90.0))

    def test_stale_streams_abort(self):
        node = FakeClockNode(sim_time=1.0)
        node.fresh = False
        with pytest.raises(episode.EpisodeAbortedError, match="stale"):
            episode.check_guards(node, 0.0, 0, make_config(episode_timeout=90.0))

    def test_no_clock_aborts(self):
        node = FakeClockNode(sim_time=None)
        with pytest.raises(episode.EpisodeAbortedError, match="clock"):
            episode.check_guards(node, 0.0, 0, make_config())


class TestWaitSimTime:
    def test_returns_when_reached(self):
        node = FakeClockNode(sim_time=0.0)

        def advance():
            time.sleep(0.05)
            node.sim_time = 2.0

        threading.Thread(target=advance).start()
        assert episode.wait_sim_time(node, target=1.0, reset_baseline=0, timeout_s=2.0) == 2.0

    def test_external_reset_aborts_wait(self):
        node = FakeClockNode(sim_time=0.0)

        def reset():
            time.sleep(0.05)
            node.resets = 1

        threading.Thread(target=reset).start()
        with pytest.raises(episode.EpisodeAbortedError, match=episode.EXTERNAL_RESET):
            episode.wait_sim_time(node, target=1.0, reset_baseline=0, timeout_s=2.0)

    def test_stalled_clock_aborts(self):
        node = FakeClockNode(sim_time=0.0)
        with pytest.raises(episode.EpisodeAbortedError, match="stalled"):
            episode.wait_sim_time(node, target=1.0, reset_baseline=0, timeout_s=0.1)


class TestNeedsResetRequest:
    def test_no_external_reset(self):
        assert episode.needs_reset_request(current_count=2, handled_count=2)

    def test_unhandled_external_reset_skips_request(self):
        assert not episode.needs_reset_request(current_count=3, handled_count=2)


class FakeResetNode:
    def __init__(self, *, subscriber=True, confirm=True):
        self.subscriber = subscriber
        self.confirm = confirm
        self.resets = 0
        self.requests = 0

    def reset_count(self):
        return self.resets

    def reset_request_has_subscriber(self):
        return self.subscriber

    def request_reset(self):
        self.requests += 1
        if self.confirm:
            self.resets += 1

    def last_reset_event_wall(self):
        return time.monotonic() - 0.1

    def clock_restarted_after(self, event_wall):
        return True


class TestEnsureFreshScene:
    def test_requests_and_confirms(self):
        node = FakeResetNode()
        handled = episode.ensure_fresh_scene(node, handled_count=0, timeout_s=1.0)
        assert node.requests == 1
        assert handled == 1

    def test_skips_request_after_external_reset(self):
        node = FakeResetNode()
        node.resets = 1  # unhandled external reset: scene already fresh
        handled = episode.ensure_fresh_scene(node, handled_count=0, timeout_s=1.0)
        assert node.requests == 0
        assert handled == 1

    def test_no_subscriber_fails(self):
        node = FakeResetNode(subscriber=False)
        with pytest.raises(RuntimeError, match="subscriber"):
            episode.ensure_fresh_scene(node, handled_count=0, timeout_s=0.3)

    def test_unconfirmed_reset_retries_then_fails(self):
        node = FakeResetNode(confirm=False)
        with pytest.raises(RuntimeError, match="not confirmed"):
            episode.ensure_fresh_scene(node, handled_count=0, timeout_s=0.4)
        assert node.requests == 2


class FakeSimNode:
    """Node whose sim clock runs at `speed` x wall clock; records publishes."""

    def __init__(self, speed=30.0):
        self._start = time.monotonic()
        self.speed = speed
        self.resets = 0
        self.published = []
        self.stopped = False

    def sim_now(self):
        return (time.monotonic() - self._start) * self.speed

    @property
    def sim_time(self):
        return self.sim_now()

    def reset_count(self):
        return self.resets

    def streams_fresh(self, max_age_s=None):
        return True

    def missing_streams(self):
        return []

    def snapshot(self):
        state = np.zeros(20, dtype=np.float32)
        images = {key: np.zeros((16, 16, 3), dtype=np.uint8) for key in ("head", "wrist_left", "wrist_right")}
        return self.sim_now(), state, images

    def publish_action(self, action, pedal_threshold):
        self.published.append(np.asarray(action))

    def publish_stop(self):
        self.stopped = True


class FakePolicy:
    """PolicyClient stand-in returning identifiable chunks."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        chunk = np.full((50, 20), float(self.calls), dtype=np.float32)
        return {"actions": chunk}


class TestRunEpisodeSync:
    def test_runs_to_timeout(self):
        node = FakeSimNode(speed=10.0)  # 1 wall-second per 10 sim-seconds
        policy = FakePolicy()
        cfg = make_config(episode_timeout=3.0, replan_steps=10)
        record = episode.run_episode_sync(node, policy, cfg)
        assert node.stopped
        # ~90 control steps in 3 sim-seconds; generous bounds for timing slop.
        assert 50 <= record["steps"] <= 95
        assert record["steps"] == len(node.published)
        assert policy.calls >= record["steps"] // 10
        assert record["num_infer_calls"] == policy.calls
        assert record["infer_ms_mean"] is not None
        assert record["duration_s"] >= 3.0


class TestRunEpisodeAsync:
    def test_runs_to_timeout_with_worker(self):
        node = FakeSimNode(speed=10.0)
        client, _ = _make_policy_client()
        worker = inference.InferenceWorker(client)
        worker.start()
        cfg = make_config(episode_timeout=3.0, replan_steps=10)
        record = episode.run_episode_async(node, worker, cfg, generation=0)
        worker.stop()
        assert node.stopped
        assert 50 <= record["steps"] <= 95
        assert record["async"] is True
        assert record["num_infer_calls"] >= 2
        assert record["queue_dry_steps"] < 10  # fake infer is near-instant
        assert "chunk_latency_steps_mean" in record

    def test_queue_dry_aborts(self):
        node = FakeSimNode(speed=30.0)

        class StallingClient:
            def infer(self, obs):
                if not hasattr(self, "first_done"):
                    self.first_done = True
                    return {"actions": np.zeros((50, 20), dtype=np.float32)}
                time.sleep(30.0)
                return {"actions": np.zeros((50, 20), dtype=np.float32)}

        worker = inference.InferenceWorker(StallingClient())
        worker.start()
        cfg = make_config(episode_timeout=60.0, replan_steps=5, queue_dry_abort_steps=10)
        with pytest.raises(episode.EpisodeAbortedError, match="dry"):
            episode.run_episode_async(node, worker, cfg, generation=0)
        assert node.stopped


def _make_policy_client():
    fake = FakePolicy()

    class Wrapper:
        def infer(self, obs):
            return fake.infer(obs)

    return Wrapper(), fake
