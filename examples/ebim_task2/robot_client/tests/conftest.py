from pathlib import Path
import sys

import pytest

# Make `import robot_client` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pytest_configure(config):
    config.addinivalue_line("markers", "ros: needs rclpy (runs inside the client container)")


def pytest_collection_modifyitems(config, items):
    try:
        import rclpy  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="rclpy not importable")
        for item in items:
            if "ros" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def free_port():
    """A TCP port on 127.0.0.1 with nothing listening on it."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def stub_server():
    """Factory for a StubServer running in a thread on a free 127.0.0.1 port; all are stopped at teardown."""
    from robot_client.stub_server import StubServer

    started = []

    def _start(**kwargs) -> "StubServer":
        server = StubServer(host="127.0.0.1", **kwargs)
        server.start()
        started.append(server)
        return server

    yield _start
    for server in started:
        server.stop()
