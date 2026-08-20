"""Policy-server access: reconnecting websocket wrapper + worker thread.

The openpi WebsocketClientPolicy is not thread-safe and a server-side
exception kills the socket, so PolicyClient rebuilds the connection on
failure, and InferenceWorker is the connection's sole owner once started
(guaranteeing at most one in-flight request -- the server is strict
lockstep per connection).
"""

import dataclasses
import logging
import queue
import threading
import time

import numpy as np


class InferenceFailedError(RuntimeError):
    """One infer call failed; the connection was rebuilt. Retry with a
    fresh observation -- never re-send a stale one."""


class FatalInferenceError(RuntimeError):
    """The reconnect budget is exhausted; the server is considered gone."""


class PolicyClient:
    """Reconnecting wrapper around openpi_client.WebsocketClientPolicy."""

    def __init__(self, host, port, api_key=None, max_reconnects: int = 3, client_factory=None):
        if client_factory is None:

            def client_factory():
                from openpi_client import websocket_client_policy

                return websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)

        self._factory = client_factory
        self.max_reconnects = max_reconnects
        self._consecutive_failures = 0
        # Blocks until the server accepts (the underlying constructor
        # retries on ConnectionRefusedError every 5 s).
        self._client = self._factory()

    def metadata(self) -> dict:
        return self._client.get_server_metadata()

    def infer(self, obs: dict) -> dict:
        try:
            result = self._client.infer(obs)
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures > self.max_reconnects:
                raise FatalInferenceError(
                    f"inference failed {self._consecutive_failures} times in a row: {exc}"
                ) from exc
            logging.warning("Inference failed (%s); reconnecting to the policy server", exc)
            self._client = self._factory()
            raise InferenceFailedError(str(exc)) from exc
        self._consecutive_failures = 0
        return result


@dataclasses.dataclass(frozen=True)
class WorkerResult:
    generation: int
    obs_step: int
    actions: np.ndarray | None  # None when error is set
    infer_ms: float
    error: str | None = None


class InferenceWorker:
    """Daemon thread that owns the PolicyClient after start().

    Lives across episodes; results are tagged with a generation (episode
    index) so responses in flight when an episode aborted are discarded.
    A failed infer emits an error WorkerResult and the thread keeps
    running (PolicyClient already reconnected); only an exhausted
    reconnect budget sets self.error and stops the thread.
    """

    _STOP = object()

    def __init__(self, policy_client: PolicyClient):
        self._client = policy_client
        self._requests = queue.Queue()
        self.results = queue.Queue()
        self.error = None  # set once by the worker thread, checked by the main loop
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference-worker")

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, generation: int, obs_step: int, obs: dict) -> None:
        self._requests.put((generation, obs_step, obs))

    def stop(self, timeout: float = 2.0) -> None:
        self._requests.put(self._STOP)
        self._thread.join(timeout)

    def purge(self) -> None:
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                return

    def drain(self, generation: int):
        """Non-blocking; yields WorkerResults for this generation,
        discarding stale-generation responses."""
        for result in self.drain_all():
            if result.generation == generation:
                yield result

    def drain_all(self):
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            item = self._requests.get()
            if item is self._STOP:
                return
            generation, obs_step, obs = item
            start = time.monotonic()
            try:
                result = self._client.infer(obs)
            except InferenceFailedError as exc:
                self.results.put(
                    WorkerResult(generation, obs_step, None, (time.monotonic() - start) * 1000.0, str(exc))
                )
                continue
            except Exception as exc:  # FatalInferenceError and anything unexpected
                self.error = exc
                self.results.put(
                    WorkerResult(generation, obs_step, None, (time.monotonic() - start) * 1000.0, str(exc))
                )
                return
            self.results.put(
                WorkerResult(
                    generation,
                    obs_step,
                    np.asarray(result["actions"]),
                    (time.monotonic() - start) * 1000.0,
                )
            )
