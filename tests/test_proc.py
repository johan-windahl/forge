"""Subprocess lifecycle guarantees used by unattended Forge runs."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path

from forge.util.proc import run, terminate_active_processes


def test_immediate_shutdown_terminates_active_command_groups(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    results = []

    def execute() -> None:
        results.append(
            run(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,time; "
                    f"pathlib.Path({str(ready)!r}).write_text('ready'); "
                    "time.sleep(30)",
                ],
                timeout=60,
            )
        )

    worker = threading.Thread(target=execute)
    worker.start()
    deadline = time.monotonic() + 3
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "test command did not start"

    terminate_active_processes(grace=0.05)
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert results and results[0].returncode != 0


def test_a_signal_that_raises_mid_wait_still_kills_the_child(tmp_path: Path) -> None:
    """The child dies even when the exception is not a subprocess timeout.

    pytest-timeout raises from a SIGALRM handler inside `communicate`, and so
    does Ctrl-C. Only `TimeoutExpired` used to reach the kill path, so those
    exceptions propagated with the process group alive. Observed for real:
    `opencode` children outliving their parent by twenty minutes.
    """
    import subprocess

    from forge.util import proc as proc_module

    ready = tmp_path / "ready"
    real_communicate = subprocess.Popen.communicate
    started: list[subprocess.Popen] = []

    def exploding_communicate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        started.append(self)
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.01)
        raise KeyboardInterrupt("simulated signal during wait")

    subprocess.Popen.communicate = exploding_communicate  # type: ignore[method-assign]
    try:
        with contextlib.suppress(KeyboardInterrupt):
            run(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,time; "
                    f"pathlib.Path({str(ready)!r}).write_text('ready'); "
                    "time.sleep(30)",
                ],
                timeout=60,
            )
    finally:
        subprocess.Popen.communicate = real_communicate  # type: ignore[method-assign]

    assert started, "the child never started"
    child = started[0]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and child.poll() is None:
        time.sleep(0.05)
    assert child.poll() is not None, "the child survived the exception"
    assert child not in proc_module._ACTIVE_PROCESSES
