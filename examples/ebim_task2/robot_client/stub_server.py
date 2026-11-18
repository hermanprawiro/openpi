"""A stand-in policy server for rehearsing the real-robot client without a GPU or a checkpoint.

Speaks the openpi websocket protocol (see `openpi_client.WebsocketClientPolicy` and
`openpi.serving.websocket_policy_server`): on connect it sends the msgpack_numpy metadata; per request it
unpacks the observation and replies `{"actions": (50, W), "server_timing": {"infer_ms": ...}}`; on an
exception it sends the traceback as a text frame and closes, like the real server.

The munich16 chunk holds every row at the request's `observation/state`, with the right-arm joint 7
(index 14) swept sinusoidally and both grippers open (1.0), so the arm visibly moves around where it is.
`--layout sim20` advertises a 20-wide sim policy instead (the client must refuse it).

Run: python -m robot_client.stub_server --port 8000 [--layout munich16|sim20] [--latency-s 0.5]
[--amplitude 0.05] [--hang-after N]

Depends only on numpy, websockets and openpi_client (not on `openpi`): it runs in the robot-client container.
"""

import argparse
import logging
import threading
import time
import traceback

import numpy as np
from openpi_client import msgpack_numpy
import websockets.exceptions
import websockets.sync.server

from robot_client import layout

LAYOUTS = ("munich16", "sim20")
SIM20_WIDTH = 20
SWEEP_PERIOD_S = 2.5
SWEEP_JOINT = 14  # right joint 7 in the munich16 action layout
_GRIPPER_OPEN = 1.0

logger = logging.getLogger(__name__)


def metadata_for(layout_id: str) -> dict:
    if layout_id == "munich16":
        return layout.POLICY_METADATA | {"config": "stub", "checkpoint": "stub"}
    if layout_id == "sim20":
        return {"config": "stub-sim20"}
    raise ValueError(f"unknown layout {layout_id!r}; expected one of {LAYOUTS}")


def make_chunk(layout_id: str, state, t_s: float, amplitude: float) -> np.ndarray:
    """The (50, W) action chunk for `state`, time-stamped `t_s` seconds since the stub started."""
    state = np.asarray(state, dtype=np.float64)
    if layout_id == "sim20":
        row = np.zeros(SIM20_WIDTH)
        n = min(state.size, SIM20_WIDTH)
        row[:n] = state.reshape(-1)[:n]
        return np.tile(row, (layout.ACTION_HORIZON, 1)).astype(np.float32)
    if state.shape != (layout.STATE_DIM,):
        raise ValueError(f"observation/state must have shape ({layout.STATE_DIM},), got {state.shape}")
    chunk = np.tile(state, (layout.ACTION_HORIZON, 1))
    chunk[:, [layout.GRIPPER_INDEX["left"], layout.GRIPPER_INDEX["right"]]] = _GRIPPER_OPEN
    row_t = t_s + np.arange(layout.ACTION_HORIZON) / layout.CONTROL_HZ
    chunk[:, SWEEP_JOINT] += amplitude * np.sin(2.0 * np.pi * row_t / SWEEP_PERIOD_S)
    return chunk.astype(np.float32)


class StubServer:
    """The stub in a background thread. `port=0` picks a free port (read `.port` after `start()`)."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        layout_id: str = "munich16",
        latency_s: float = 0.5,
        amplitude: float = 0.05,
        hang_after: int | None = None,
    ):
        self._metadata = metadata_for(layout_id)
        self._host, self._requested_port = host, port
        self._layout_id = layout_id
        self._latency_s = latency_s
        self._amplitude = amplitude
        self._hang_after = hang_after
        self._lock = threading.Lock()
        self._n_requests = 0  # across connections: once the server hangs, it hangs for reconnects too
        self._connections: set = set()
        self._t0 = time.monotonic()
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    def start(self) -> None:
        self._server = websockets.sync.server.serve(
            self._handler, self._host, self._requested_port, compression=None, max_size=None
        )
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="stub-policy-server")
        self._thread.start()

    def wait(self) -> None:
        """Block until the server thread ends (i.e. forever, until `stop()`)."""
        self._thread.join()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        with self._lock:
            connections = list(self._connections)
        for ws in connections:
            ws.close()
        self._thread.join(5)
        self._server = None

    def _handler(self, ws) -> None:
        packer = msgpack_numpy.Packer()
        with self._lock:
            self._connections.add(ws)
        try:
            ws.send(packer.pack(self._metadata))
            while True:
                try:
                    message = ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    return
                with self._lock:
                    self._n_requests += 1
                    hanging = self._hang_after is not None and self._n_requests > self._hang_after
                if hanging:
                    continue  # read and ignore: the client's receive timeout is what is being exercised
                try:
                    start = time.monotonic()
                    obs = msgpack_numpy.unpackb(message)
                    time.sleep(self._latency_s)
                    reply = {
                        "actions": make_chunk(
                            self._layout_id, obs["observation/state"], start - self._t0, self._amplitude
                        ),
                        "server_timing": {"infer_ms": (time.monotonic() - start) * 1000.0},
                    }
                    ws.send(packer.pack(reply))
                except websockets.exceptions.ConnectionClosed:
                    return
                except Exception:
                    ws.send(traceback.format_exc())
                    ws.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
                    return
        finally:
            with self._lock:
                self._connections.discard(ws)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--layout", choices=LAYOUTS, default="munich16")
    parser.add_argument("--latency-s", type=float, default=0.5, help="simulated inference time per request")
    parser.add_argument("--amplitude", type=float, default=0.05, help="right joint 7 sweep amplitude (rad)")
    parser.add_argument("--hang-after", type=int, default=None, help="stop replying after N requests")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = StubServer(args.host, args.port, args.layout, args.latency_s, args.amplitude, args.hang_after)
    server.start()
    logger.info(
        "stub policy server (%s) listening on %s:%d, latency %.2f s, hang-after %s",
        args.layout,
        args.host,
        server.port,
        args.latency_s,
        args.hang_after,
    )
    try:
        server.wait()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
