"""The client's link to the policy server: bounded connect and receive, re-validated on every connection.

`openpi_client.WebsocketClientPolicy` retries a refused connection forever, has no receive timeout, and lets
other connect errors (DNS, handshake, timeouts) escape `PolicyClient`'s reconnect and kill the
`InferenceWorker` for good. `RobotPolicyConnection` speaks the same protocol but every network wait is
bounded and every failure to connect is one clear `PolicyUnreachableError`. The server metadata is checked
against the munich16 layout on *every* connection, so a server swapped for another checkpoint between two
requests is caught at the reconnect (`LayoutMismatchError`, fatal for the run).

Depends on `openpi_client`, `websockets` and the `eval_client` worker; no ROS.
"""

import logging
import time

from eval_client.inference import InferenceWorker
from eval_client.inference import PolicyClient
from openpi_client import msgpack_numpy
import websockets.sync.client

from robot_client import layout
from robot_client.config import SiteConfig
from robot_client.layout import LayoutMismatchError

logger = logging.getLogger(__name__)


class PolicyUnreachableError(ConnectionError):
    """The policy server could not be connected to (or did not introduce itself) in time."""


class RobotPolicyConnection:
    """One websocket connection to a munich16 policy server (replaced, not repaired, on any failure)."""

    def __init__(
        self,
        host: str,
        port: int | None,
        api_key: str | None,
        prompt: str,
        connect_timeout_s: float,
        infer_timeout_s: float,
    ):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._target = f"{host}:{port}" if port is not None else host
        self._connect_timeout_s = connect_timeout_s
        self._infer_timeout_s = infer_timeout_s
        self._headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._ws, self._metadata = self._connect()
        try:
            layout.validate_server_metadata(self._metadata, prompt)
        except LayoutMismatchError:
            self.close()
            raise

    def get_server_metadata(self) -> dict:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        try:
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=self._infer_timeout_s)
        except BaseException:
            # A late reply must never be taken for the answer to a later request.
            self.close()
            raise
        if isinstance(response, str):
            # The server sends its traceback as text and closes the connection.
            self.close()
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # best effort on an already broken socket
                logger.debug("closing the policy connection failed", exc_info=True)

    def _connect(self):
        deadline = time.monotonic() + self._connect_timeout_s
        ws = None
        try:
            ws = websockets.sync.client.connect(
                self._uri,
                open_timeout=self._connect_timeout_s,
                additional_headers=self._headers,
                compression=None,
                max_size=None,
            )
            first = ws.recv(timeout=max(deadline - time.monotonic(), 0.001))
            if isinstance(first, str):
                raise RuntimeError(f"server sent an error instead of its metadata:\n{first}")
            return ws, msgpack_numpy.unpackb(first)
        except Exception as exc:
            if ws is not None:
                ws.close()
            reason = str(exc) or type(exc).__name__
            if not isinstance(exc, OSError | TimeoutError):
                reason = f"{type(exc).__name__}: {reason}"
            raise PolicyUnreachableError(
                f"cannot reach policy server at {self._target} within {self._connect_timeout_s:g}s: {reason}"
            ) from exc


def make_policy_client(cfg: SiteConfig) -> PolicyClient:
    """A `PolicyClient` whose every (re)connection is a validated, timeout-bounded `RobotPolicyConnection`."""
    server = cfg.server

    def factory() -> RobotPolicyConnection:
        return RobotPolicyConnection(
            host=server.host,
            port=server.port,
            api_key=server.api_key,
            prompt=cfg.policy.prompt,
            connect_timeout_s=server.connect_timeout_s,
            infer_timeout_s=server.infer_timeout_s,
        )

    return PolicyClient(server.host, server.port, server.api_key, client_factory=factory)


def ensure_worker(worker: InferenceWorker | None, cfg: SiteConfig) -> InferenceWorker:
    """`worker` if its thread is alive, else a fresh started worker on a new connection.

    Called before every episode, so a worker killed by an exhausted reconnect budget (or a server that went
    away mid-episode) is replaced by the next episode's reconnect. Raises `PolicyUnreachableError` if the
    server cannot be reached and `LayoutMismatchError` (fatal for the run) if it is not a munich16 policy.
    """
    if worker is not None and worker.is_alive():
        return worker
    if worker is not None:
        logger.warning("inference worker is dead (%s); reconnecting to the policy server", worker.error)
    fresh = InferenceWorker(make_policy_client(cfg))
    fresh.start()
    return fresh
