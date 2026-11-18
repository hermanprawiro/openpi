import math
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from robot_client import layout
from robot_client import stub_server
import websockets.exceptions
import websockets.sync.client

ROBOT_CLIENT_PARENT = Path(stub_server.__file__).resolve().parents[1]


def obs_with_state(state):
    return {"observation/state": np.asarray(state, dtype=np.float64), "prompt": layout.PROMPT}


def connect(server):
    ws = websockets.sync.client.connect(f"ws://127.0.0.1:{server.port}", compression=None, max_size=None)
    return ws, msgpack_numpy.unpackb(ws.recv(timeout=2))


def request(ws, obs):
    ws.send(msgpack_numpy.Packer().pack(obs))
    return ws.recv(timeout=5)


def test_munich16_metadata_is_the_layout_plus_stub_markers(stub_server):
    ws, meta = connect(stub_server())
    with ws:
        assert meta == layout.POLICY_METADATA | {"config": "stub", "checkpoint": "stub"}
        layout.validate_server_metadata(meta, layout.PROMPT)


def test_sim20_metadata_is_not_a_munich16_policy(stub_server):
    ws, meta = connect(stub_server(layout_id="sim20"))
    with ws:
        assert meta == {"config": "stub-sim20"}
        with pytest.raises(layout.LayoutMismatchError):
            layout.validate_server_metadata(meta, layout.PROMPT)


def test_munich16_chunk_follows_the_state_with_a_right_joint_7_sweep(stub_server):
    server = stub_server(amplitude=0.05)
    state = np.linspace(-0.5, 0.5, 16)
    ws, _ = connect(server)
    with ws:
        reply = msgpack_numpy.unpackb(request(ws, obs_with_state(state)))
    actions = reply["actions"]
    assert actions.shape == (layout.ACTION_HORIZON, layout.ACTION_DIM)
    assert reply["server_timing"]["infer_ms"] >= 0
    expected = np.tile(state, (layout.ACTION_HORIZON, 1))
    expected[:, [7, 15]] = 1.0  # grippers
    deviation = np.abs(actions - expected)
    assert deviation[:, 14].max() <= 0.05 + 1e-6
    assert deviation[:, 14].max() > 0.01  # it really moves
    deviation[:, 14] = 0.0
    assert deviation.max() < 1e-6


def test_sim20_chunk_is_20_wide(stub_server):
    ws, _ = connect(stub_server(layout_id="sim20"))
    with ws:
        reply = msgpack_numpy.unpackb(request(ws, obs_with_state(np.zeros(20))))
    assert reply["actions"].shape == (layout.ACTION_HORIZON, 20)


def test_latency_is_applied(stub_server):
    ws, _ = connect(stub_server(latency_s=0.3))
    with ws:
        start = time.monotonic()
        reply = msgpack_numpy.unpackb(request(ws, obs_with_state(np.zeros(16))))
    assert time.monotonic() - start >= 0.3
    assert reply["server_timing"]["infer_ms"] >= 290


def test_bad_request_gets_a_traceback_string_like_the_openpi_server(stub_server):
    ws, _ = connect(stub_server())
    with ws:
        reply = request(ws, obs_with_state(np.zeros(3)))  # wrong state width
        assert isinstance(reply, str)
        assert "Traceback" in reply and "state" in reply
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            ws.recv(timeout=2)


def test_hang_after_stops_replying_but_keeps_accepting_connections(stub_server):
    server = stub_server(hang_after=1)
    ws, _ = connect(server)
    with ws:
        assert isinstance(request(ws, obs_with_state(np.zeros(16))), bytes)
        with pytest.raises(TimeoutError):
            ws.send(msgpack_numpy.Packer().pack(obs_with_state(np.zeros(16))))
            ws.recv(timeout=0.5)
    ws2, meta = connect(server)  # a reconnect still gets metadata, and then hangs again
    with ws2:
        assert meta["layout"] == layout.LAYOUT_ID
        with pytest.raises(TimeoutError):
            ws2.send(msgpack_numpy.Packer().pack(obs_with_state(np.zeros(16))))
            ws2.recv(timeout=0.5)


def test_stop_closes_open_connections_and_frees_the_port(stub_server):
    server = stub_server()
    port = server.port
    ws, _ = connect(server)
    server.stop()
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        ws.recv(timeout=2)
    with pytest.raises(ConnectionRefusedError), socket.create_connection(("127.0.0.1", port), timeout=2):
        pass


def test_parse_args_defaults_and_flags():
    args = stub_server.parse_args([])
    assert (args.port, args.layout, args.latency_s, args.amplitude, args.hang_after) == (
        8000,
        "munich16",
        0.5,
        0.05,
        None,
    )
    args = stub_server.parse_args(
        ["--port", "9001", "--layout", "sim20", "--latency-s", "0", "--amplitude", "0.1", "--hang-after", "3"]
    )
    assert (args.port, args.layout, args.latency_s, args.amplitude, args.hang_after) == (9001, "sim20", 0.0, 0.1, 3)
    with pytest.raises(SystemExit):
        stub_server.parse_args(["--layout", "nope"])


def test_stub_does_not_import_openpi():
    code = "import robot_client.stub_server, sys; sys.exit(1 if 'openpi' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROBOT_CLIENT_PARENT, check=False, timeout=30)
    assert result.returncode == 0


def test_module_entry_point_serves(free_port):
    proc = subprocess.Popen(
        [sys.executable, "-m", "robot_client.stub_server", "--port", str(free_port), "--host", "127.0.0.1"],
        cwd=ROBOT_CLIENT_PARENT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        meta = None
        while time.monotonic() < deadline and meta is None:
            try:
                with websockets.sync.client.connect(f"ws://127.0.0.1:{free_port}", open_timeout=1) as ws:
                    meta = msgpack_numpy.unpackb(ws.recv(timeout=2))
            except (OSError, TimeoutError):
                time.sleep(0.1)
        assert meta is not None and meta["layout"] == layout.LAYOUT_ID
        assert math.isfinite(meta["control_hz"])
    finally:
        proc.terminate()
        proc.wait(timeout=10)
