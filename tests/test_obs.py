"""Logging must not be able to kill the run it is describing."""

from __future__ import annotations

import io
from pathlib import Path

from forge.obs.log import _Sink


def _sink(**kwargs) -> _Sink:  # type: ignore[no-untyped-def]
    defaults = dict(
        path=None,
        console=None,
        console_level="info",
        file_level="info",
        max_bytes=0,
        backups=1,
        color=False,
    )
    defaults.update(kwargs)
    return _Sink(**defaults)  # type: ignore[arg-type]


def _record(msg: str = "hello") -> dict:
    return {"ts": "2026-07-29T12:00:00.000Z", "level": "info", "msg": msg}


def test_a_closed_console_does_not_raise(tmp_path: Path) -> None:
    """The console can go away underneath a long run: a detached terminal, a
    closed pipe, a captured stream torn down. Losing the pretty output is not a
    reason to take down a build that is otherwise healthy."""
    stream = io.StringIO()
    sink = _sink(console=stream)
    sink.emit(_record("before"))
    stream.close()

    sink.emit(_record("after"))  # must not raise
    assert sink.console is None  # and is not retried on every subsequent record


def test_a_dead_file_handle_does_not_raise(tmp_path: Path) -> None:
    """Same for the JSONL sink: a full disk or unmounted volume degrades to no
    file logging, not to a crashed orchestrator."""
    path = tmp_path / "logs" / "forge.jsonl"
    sink = _sink(path=path)
    sink.emit(_record("first"))
    sink._handle.close()

    sink.emit(_record("second"))
    assert sink._handle is None
    assert "first" in path.read_text(encoding="utf-8")


def test_the_console_survives_normal_use(tmp_path: Path) -> None:
    """The defensive path must not have swallowed the happy one."""
    stream = io.StringIO()
    sink = _sink(console=stream)
    sink.emit(_record("visible"))
    assert "visible" in stream.getvalue()
    assert sink.console is stream
