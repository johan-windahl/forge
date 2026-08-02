"""Structured logging.

Forge runs unattended for days; the log is the only witness to what happened.
Two consumers with opposite needs are served from one call site:

* A **human** tailing the console wants short, coloured, scannable lines.
* A **program** -- the retrospective analyser, the dashboard, a future incident
  triage -- wants machine-parseable records with stable field names.

So every log call emits JSON Lines to a rotating file and a compact rendering to
the console. Contextual fields (project, node, agent, attempt) are bound once
via :meth:`Logger.bind` and then attach automatically to every downstream
record, which is what makes it possible to reconstruct a single node's story out
of an interleaved multi-worker log.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from ..util.clock import iso_ms

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40, "fatal": 50}

_COLORS = {
    "debug": "\033[2;37m",
    "info": "\033[0;36m",
    "warn": "\033[0;33m",
    "error": "\033[0;31m",
    "fatal": "\033[1;41;37m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"


class _Sink:
    """Writes JSON Lines to a size-rotated file, and a rendering to a stream."""

    def __init__(
        self,
        *,
        path: Path | None,
        console: TextIO | None,
        console_level: str,
        file_level: str,
        max_bytes: int,
        backups: int,
        color: bool,
    ) -> None:
        self.path = path
        self.console = console
        self.console_level = LEVELS[console_level]
        self.file_level = LEVELS[file_level]
        self.max_bytes = max_bytes
        self.backups = backups
        self.color = color
        self._lock = threading.Lock()
        self._handle: TextIO | None = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        """Write a record. Never raises.

        A logger that can kill the process it is describing is worse than no
        logger. Both sinks can legitimately go away underneath a long run -- the
        console when a terminal or a captured stream is closed, the file when its
        directory is unmounted or a disk fills -- and neither is a reason to take
        down a build that is otherwise fine. The console is dropped for good once
        it fails, because a stream that has closed does not reopen and retrying
        per record would turn every log line into an exception.
        """
        level = LEVELS.get(record["level"], 20)
        with self._lock:
            if self._handle and level >= self.file_level:
                try:
                    self._handle.write(json.dumps(record, default=str) + "\n")
                    self._handle.flush()
                    self._maybe_rotate()
                except (OSError, ValueError):
                    self._handle = None
            if self.console and level >= self.console_level:
                try:
                    self.console.write(self._render(record) + "\n")
                    self.console.flush()
                except (OSError, ValueError):
                    self.console = None

    def _maybe_rotate(self) -> None:
        if not (self._handle and self.path and self.max_bytes):
            return
        try:
            if self._handle.tell() < self.max_bytes:
                return
        except OSError:  # pragma: no cover
            return
        self._handle.close()
        for i in range(self.backups - 1, 0, -1):
            src, dst = self.path.with_suffix(f".{i}.jsonl"), self.path.with_suffix(f".{i + 1}.jsonl")
            if src.exists():
                src.replace(dst)
        self.path.replace(self.path.with_suffix(".1.jsonl"))
        self._handle = self.path.open("a", encoding="utf-8")

    def _render(self, record: dict[str, Any]) -> str:
        level = record["level"]
        head = f"{level.upper():<5}"
        if self.color:
            head = f"{_COLORS.get(level, '')}{head}{_RESET}"
        ts = record["ts"][11:23]
        scope = record.get("node") or record.get("agent") or record.get("component") or ""
        parts = [f"{_DIM if self.color else ''}{ts}{_RESET if self.color else ''}", head]
        if scope:
            parts.append(f"[{scope}]")
        parts.append(record["msg"])
        extras = {
            k: v
            for k, v in record.items()
            if k not in ("ts", "level", "msg", "node", "agent", "component", "logger")
        }
        if extras:
            rendered = " ".join(f"{k}={_compact(v)}" for k, v in sorted(extras.items()))
            parts.append(f"{_DIM if self.color else ''}{rendered}{_RESET if self.color else ''}")
        return " ".join(parts)

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._handle.close()
                self._handle = None


def _compact(value: Any, limit: int = 120) -> str:
    if isinstance(value, str):
        text = value if len(value) <= limit else value[: limit - 1] + "…"
        return text if " " not in text else json.dumps(text)
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


_sink: _Sink | None = None
# Reentrant: the lazy-initialisation path calls setup_logging while already
# holding this lock, and a plain Lock would deadlock the first log call made
# before setup_logging ran.
_sink_lock = threading.RLock()


def setup_logging(
    *,
    path: Path | None = None,
    console: bool = True,
    console_level: str = "info",
    file_level: str = "debug",
    max_bytes: int = 64 * 1024 * 1024,
    backups: int = 5,
    color: bool | None = None,
) -> None:
    """Install the process-wide sink. Safe to call again to reconfigure."""
    global _sink
    if color is None:
        color = console and sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    with _sink_lock:
        if _sink:
            _sink.close()
        _sink = _Sink(
            path=path,
            console=sys.stderr if console else None,
            console_level=console_level,
            file_level=file_level,
            max_bytes=max_bytes,
            backups=backups,
            color=color,
        )


def _ensure_sink() -> _Sink:
    """Return the sink, installing a console-only default on first use.

    Library code logs freely; nothing may require the application to have called
    ``setup_logging`` first, and nothing may deadlock if it has not.
    """
    sink = _sink
    if sink is not None:
        return sink
    with _sink_lock:
        if _sink is None:
            setup_logging()
        assert _sink is not None
        return _sink


class Logger:
    """A logger with bound context. Immutable: ``bind`` returns a new logger."""

    __slots__ = ("_context", "_name")

    def __init__(self, name: str, context: dict[str, Any] | None = None) -> None:
        self._name = name
        self._context = context or {}

    def bind(self, **fields: Any) -> Logger:
        return Logger(self._name, {**self._context, **fields})

    @contextmanager
    def scope(self, **fields: Any) -> Iterator[Logger]:
        """Temporarily bind fields for a block of work."""
        yield self.bind(**fields)

    def _log(self, level: str, msg: str, **fields: Any) -> None:
        sink = _ensure_sink()
        record = {"ts": iso_ms(), "level": level, "logger": self._name, "msg": msg}
        record.update(self._context)
        record.update(fields)
        sink.emit(record)

    def debug(self, msg: str, **f: Any) -> None:
        self._log("debug", msg, **f)

    def info(self, msg: str, **f: Any) -> None:
        self._log("info", msg, **f)

    def warn(self, msg: str, **f: Any) -> None:
        self._log("warn", msg, **f)

    def error(self, msg: str, **f: Any) -> None:
        self._log("error", msg, **f)

    def fatal(self, msg: str, **f: Any) -> None:
        self._log("fatal", msg, **f)

    def exception(self, msg: str, exc: BaseException, **f: Any) -> None:
        """Log an error with the exception type and message, no stack spam.

        Full tracebacks go to the ledger as failure events where they can be
        retrieved on demand; the log stays readable.
        """
        from ..errors import ForgeError

        detail: dict[str, Any] = {"error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(exc, ForgeError):
            detail["retryable"] = exc.retryable
            detail["transient"] = exc.transient
        self._log("error", msg, **detail, **f)


def get_logger(name: str, **context: Any) -> Logger:
    return Logger(name, context or None)
