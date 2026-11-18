"""Operator keys for the episode runner: Enter (start/resume), `p` (pause), `q` (end episode).

`poll()` never blocks. `TerminalKeys` puts a TTY into cbreak mode (single key presses, no Enter needed for
`p`/`q`) and restores the saved terminal attributes on `close()`; on anything that is not a TTY (a pipe,
a file, an `io.StringIO`) it reads whole lines and reports each line's first character (an empty line is
Enter). Keys are lower-cased; Enter is `ENTER`.
"""

from collections.abc import Mapping
import io
import os
import select
import sys
from typing import Protocol

ENTER = "\n"


class KeySource(Protocol):
    def poll(self) -> str | None:
        """The next pending key, or None (never blocks)."""


class FakeKeys:
    """For tests: `script[n]` is returned by the n-th `poll()` (0-based, i.e. at runner tick n)."""

    def __init__(self, script: Mapping[int, str]):
        self._script = dict(script)
        self._polls = 0

    def poll(self) -> str | None:
        key = self._script.get(self._polls)
        self._polls += 1
        return key

    def close(self) -> None:
        pass


def _line_key(line: str) -> str:
    text = line.strip()
    return text[0].lower() if text else ENTER


class TerminalKeys:
    def __init__(self, stream=None):
        self._stream = sys.stdin if stream is None else stream
        self._buf = ""
        self._eof = False
        self._saved = None
        try:
            self._fd: int | None = self._stream.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            self._fd = None
        if self._fd is not None and os.isatty(self._fd):
            import termios
            import tty

            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    @property
    def cbreak(self) -> bool:
        return self._saved is not None

    def poll(self) -> str | None:
        if self.cbreak:
            if not self._readable():
                return None
            ch = os.read(self._fd, 1).decode(errors="replace")
            if not ch:
                return None
            return ENTER if ch in "\r\n" else ch.lower()
        return self._poll_line()

    def _readable(self) -> bool:
        ready, _, _ = select.select([self._fd], [], [], 0)
        return bool(ready)

    def _poll_line(self) -> str | None:
        if "\n" not in self._buf and not self._eof:
            if self._fd is None:
                self._buf += self._stream.readline()  # in-memory stream: never blocks
            else:
                # Read the fd directly: a buffered text stream can hold lines that select() does not see.
                while self._readable():
                    data = os.read(self._fd, 4096)
                    if not data:
                        self._eof = True
                        break
                    self._buf += data.decode(errors="replace")
        if "\n" not in self._buf:
            return None
        line, self._buf = self._buf.split("\n", 1)
        return _line_key(line)

    def close(self) -> None:
        """Restore the terminal attributes saved at construction (idempotent)."""
        if self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def __enter__(self) -> "TerminalKeys":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
