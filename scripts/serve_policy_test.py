import socket

from . import serve_policy


def test_local_ip_survives_an_unresolvable_hostname(monkeypatch):
    """Offline hosts may not resolve their own hostname; the address is only logged, so the server must still start."""

    def fail(name):
        raise socket.gaierror(-3, "Temporary failure in name resolution")

    monkeypatch.setattr(socket, "gethostbyname", fail)
    assert serve_policy._local_ip("isolated-host") == "unresolved"  # noqa: SLF001


def test_local_ip_resolves_localhost():
    assert serve_policy._local_ip("localhost").startswith("127.")  # noqa: SLF001
