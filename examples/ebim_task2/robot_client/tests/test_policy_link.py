from pathlib import Path
import socket
import threading
import time

from eval_client.inference import InferenceWorker
from eval_client.inference import PolicyClient
import numpy as np
import pytest
from robot_client import config as config_mod
from robot_client import layout
from robot_client.config import load_site_config
from robot_client.layout import LayoutMismatchError
from robot_client.policy_link import PolicyUnreachableError
from robot_client.policy_link import RobotPolicyConnection
from robot_client.policy_link import ensure_worker
from robot_client.policy_link import make_policy_client

MUNICH_YAML = Path(config_mod.__file__).with_name("robot_site.munich.yaml")
TIMEOUT_S = 1.0


def site_config(port, host="127.0.0.1", api_key=None):
    env = {"POLICY_SERVER_HOST": host, "POLICY_SERVER_PORT": str(port)}
    if api_key:
        env["POLICY_SERVER_API_KEY"] = api_key
    return load_site_config(
        MUNICH_YAML, [f"server.connect_timeout_s={TIMEOUT_S}", f"server.infer_timeout_s={TIMEOUT_S}"], env
    )


def make_obs(state=None):
    state = np.zeros(layout.STATE_DIM) if state is None else state
    return {"observation/state": np.asarray(state, dtype=np.float64), "prompt": layout.PROMPT}


def connection(port, **kw):
    kw = {"host": "127.0.0.1", "api_key": None, "prompt": layout.PROMPT} | kw
    return RobotPolicyConnection(port=port, connect_timeout_s=TIMEOUT_S, infer_timeout_s=TIMEOUT_S, **kw)


def test_make_policy_client_validates_metadata_and_infers(stub_server):
    server = stub_server(latency_s=0.0, amplitude=0.05)
    client = make_policy_client(site_config(server.port))
    assert isinstance(client, PolicyClient)
    meta = client.metadata()
    assert meta["layout"] == layout.LAYOUT_ID and meta["config"] == "stub"
    state = np.linspace(-1.0, 1.0, layout.STATE_DIM)
    result = client.infer(make_obs(state))
    assert result["actions"].shape == (layout.ACTION_HORIZON, layout.ACTION_DIM)
    assert np.abs(result["actions"][0] - state)[[i for i in range(16) if i not in (7, 15)]].max() <= 0.05 + 1e-6


def test_sim20_server_is_refused_at_connect(stub_server):
    server = stub_server(layout_id="sim20")
    with pytest.raises(LayoutMismatchError):
        make_policy_client(site_config(server.port))


def test_prompt_mismatch_is_refused(stub_server):
    server = stub_server()
    with pytest.raises(LayoutMismatchError, match="prompt"):
        connection(server.port, prompt="Do something else")


def test_nothing_listening_raises_unreachable_within_the_connect_timeout(free_port):
    start = time.monotonic()
    with pytest.raises(PolicyUnreachableError) as exc:
        make_policy_client(site_config(free_port))
    assert time.monotonic() - start < TIMEOUT_S + 1.0
    message = str(exc.value)
    assert f"cannot reach policy server at 127.0.0.1:{free_port} within 1s" in message


def test_server_that_accepts_but_never_speaks_times_out(free_port):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", free_port))
        listener.listen(8)  # the kernel completes TCP handshakes; nobody ever answers the websocket upgrade
        start = time.monotonic()
        with pytest.raises(PolicyUnreachableError, match="cannot reach policy server"):
            make_policy_client(site_config(free_port))
        assert time.monotonic() - start < TIMEOUT_S + 1.0


def test_unresolvable_host_is_unreachable():
    start = time.monotonic()
    with pytest.raises(PolicyUnreachableError, match="cannot reach policy server at no-such-host.invalid:8000"):
        connection(8000, host="no-such-host.invalid")
    assert time.monotonic() - start < TIMEOUT_S + 3.0


def test_api_key_is_sent_as_an_authorization_header(free_port):
    seen = {}
    ready = threading.Event()

    def serve_once():
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", free_port))
            listener.listen(1)
            ready.set()
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2)
                seen["request"] = conn.recv(4096).decode("latin-1")

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    assert ready.wait(2)
    with pytest.raises(PolicyUnreachableError):
        connection(free_port, api_key="s3cret")  # the fake server never completes the handshake
    thread.join(3)
    assert "authorization: api-key s3cret" in seen["request"].lower()


def test_unreachable_message_never_contains_the_api_key(free_port):
    with pytest.raises(PolicyUnreachableError) as exc:
        connection(free_port, api_key="s3cret")
    assert "s3cret" not in str(exc.value)


def test_server_traceback_becomes_runtime_error(stub_server):
    conn = connection(stub_server().port)
    with pytest.raises(RuntimeError, match="Error in inference server"):
        conn.infer(make_obs(np.zeros(3)))  # the stub rejects a wrong-width state


def test_hung_server_fails_the_infer_in_time_and_the_worker_survives(stub_server):
    server = stub_server(latency_s=0.0, hang_after=1)
    worker = InferenceWorker(make_policy_client(site_config(server.port)))
    worker.start()
    try:
        worker.submit(0, 0, make_obs())
        first = worker.results.get(timeout=TIMEOUT_S + 2)
        assert first.error is None and first.actions.shape == (layout.ACTION_HORIZON, layout.ACTION_DIM)

        start = time.monotonic()
        worker.submit(0, 1, make_obs())
        second = worker.results.get(timeout=TIMEOUT_S + 1.0)
        assert time.monotonic() - start < TIMEOUT_S + 1.0
        assert second.actions is None and second.error
        assert worker.is_alive() and worker.error is None
    finally:
        worker.stop()


def test_restart_as_sim20_is_caught_on_reconnect(stub_server):
    server = stub_server(latency_s=0.0)
    port = server.port
    client = make_policy_client(site_config(port))
    assert client.infer(make_obs())["actions"].shape[1] == layout.ACTION_DIM
    server.stop()
    stub_server(layout_id="sim20", port=port)
    with pytest.raises(LayoutMismatchError):
        client.infer(make_obs())


def test_ensure_worker_builds_reuses_and_replaces(stub_server):
    server = stub_server(latency_s=0.0)
    cfg = site_config(server.port)
    first = ensure_worker(None, cfg)
    try:
        assert first.is_alive()
        assert ensure_worker(first, cfg) is first
        first.stop()
        assert not first.is_alive()
        second = ensure_worker(first, cfg)
        assert second is not first and second.is_alive()
        second.submit(0, 0, make_obs())
        assert second.results.get(timeout=TIMEOUT_S + 2).error is None
        second.stop()
    finally:
        first.stop()


def test_ensure_worker_is_fatal_for_a_wrong_layout_server(stub_server):
    cfg = site_config(stub_server(layout_id="sim20").port)
    with pytest.raises(LayoutMismatchError):
        ensure_worker(None, cfg)


def test_ensure_worker_unreachable_server_raises_unreachable(free_port):
    with pytest.raises(PolicyUnreachableError):
        ensure_worker(None, site_config(free_port))
