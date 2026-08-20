import time

from eval_client import inference
import numpy as np
import pytest


class FakeWebsocketPolicy:
    """Stands in for WebsocketClientPolicy: scripted failures."""

    def __init__(self, fail_next: list):
        self.fail_next = fail_next
        self.calls = 0

    def get_server_metadata(self):
        return {"fake": True}

    def infer(self, obs):
        self.calls += 1
        if self.fail_next and self.fail_next.pop(0):
            raise RuntimeError("connection lost")
        return {"actions": np.zeros((50, 20), dtype=np.float32)}


def make_client(fail_script, max_reconnects=3):
    instances = []

    def factory():
        instance = FakeWebsocketPolicy(fail_script)
        instances.append(instance)
        return instance

    client = inference.PolicyClient("localhost", 8000, max_reconnects=max_reconnects, client_factory=factory)
    return client, instances


class TestPolicyClient:
    def test_infer_passthrough(self):
        client, _ = make_client([])
        result = client.infer({"prompt": "x"})
        assert result["actions"].shape == (50, 20)

    def test_failure_reconnects_and_raises(self):
        client, instances = make_client([True])
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})
        assert len(instances) == 2  # original + reconnect

    def test_budget_exhaustion_is_fatal(self):
        client, _ = make_client([True] * 5, max_reconnects=2)
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})
        with pytest.raises(inference.FatalInferenceError):
            client.infer({})

    def test_success_resets_budget(self):
        client, _ = make_client([True, False, True, False, True], max_reconnects=2)
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})
        client.infer({})  # success resets the failure count
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})
        client.infer({})
        with pytest.raises(inference.InferenceFailedError):
            client.infer({})

    def test_metadata(self):
        client, _ = make_client([])
        assert client.metadata() == {"fake": True}


def wait_for_results(worker, count, timeout=2.0):
    deadline = time.monotonic() + timeout
    collected = []
    while len(collected) < count and time.monotonic() < deadline:
        collected.extend(worker.drain_all())
        time.sleep(0.005)
    return collected


class TestInferenceWorker:
    def _worker(self, fail_script, max_reconnects=3):
        client, _ = make_client(fail_script, max_reconnects)
        worker = inference.InferenceWorker(client)
        worker.start()
        return worker

    def test_result_tagging(self):
        worker = self._worker([])
        worker.submit(generation=1, obs_step=5, obs={})
        results = wait_for_results(worker, 1)
        worker.stop()
        assert len(results) == 1
        assert results[0].generation == 1
        assert results[0].obs_step == 5
        assert results[0].error is None
        assert results[0].actions.shape == (50, 20)
        assert results[0].infer_ms >= 0.0

    def test_failed_infer_keeps_worker_alive(self):
        worker = self._worker([True, False])
        worker.submit(0, 0, {})
        failed = wait_for_results(worker, 1)
        assert failed[0].error is not None
        assert failed[0].actions is None
        assert worker.is_alive()
        worker.submit(0, 1, {})
        ok = wait_for_results(worker, 1)
        worker.stop()
        assert ok[0].error is None

    def test_fatal_error_stops_worker(self):
        worker = self._worker([True] * 10, max_reconnects=1)
        worker.submit(0, 0, {})  # first failure: within budget, worker survives
        worker.submit(0, 1, {})  # second consecutive failure: budget exhausted, fatal
        wait_for_results(worker, 2)
        deadline = time.monotonic() + 2.0
        while worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not worker.is_alive()
        assert worker.error is not None

    def test_drain_discards_stale_generations(self):
        worker = self._worker([])
        worker.submit(generation=0, obs_step=0, obs={})
        worker.submit(generation=1, obs_step=3, obs={})
        deadline = time.monotonic() + 2.0
        fresh = []
        while not fresh and time.monotonic() < deadline:
            fresh.extend(worker.drain(generation=1))
            time.sleep(0.005)
        worker.stop()
        assert len(fresh) == 1
        assert fresh[0].obs_step == 3

    def test_purge_empties_results(self):
        worker = self._worker([])
        worker.submit(0, 0, {})
        deadline = time.monotonic() + 2.0
        while worker.results.qsize() == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        worker.purge()
        assert list(worker.drain(0)) == []
        worker.stop()

    def test_stop_joins_promptly(self):
        worker = self._worker([])
        start = time.monotonic()
        worker.stop()
        assert time.monotonic() - start < 1.0
        assert not worker.is_alive()
