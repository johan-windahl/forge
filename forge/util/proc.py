"""Subprocess execution with hard guarantees.

An autonomous system runs untrusted-ish commands for days. Three failure modes
sink naive ``subprocess.run`` usage:

1. **Orphaned children.** A build spawns a daemon; killing the parent leaves it
   holding a port. Forge puts every command in its own process *group* and kills
   the group.
2. **Unbounded output.** A test loop printing to stdout fills RAM. Output is
   capped with head/tail retention -- the two regions that actually carry
   diagnostic value.
3. **Hangs.** Timeouts are always set, never optional.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import CommandTimeout, SandboxError

DEFAULT_OUTPUT_LIMIT = 256 * 1024

# Commands own separate process groups so their descendants can be killed on a
# timeout. Keep the live roots as well: an immediate orchestrator shutdown must
# explicitly terminate those groups because they are outside its own group.
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


def terminate_active_processes(*, grace: float = 0.5) -> None:
    """Terminate every command still owned by this Forge process."""
    with _ACTIVE_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for proc in active:
        _kill_group(proc, grace=grace)


@dataclass(slots=True)
class ProcResult:
    """The outcome of one command execution."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    truncated: bool = False
    cwd: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout}\n--- stderr ---\n{self.stderr}"
        return self.stdout or self.stderr

    def tail(self, lines: int = 60) -> str:
        """Last ``lines`` lines of combined output -- what a model needs to debug."""
        return "\n".join(self.combined.splitlines()[-lines:])

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "duration": round(self.duration, 3),
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "cwd": self.cwd,
        }


def clamp_output(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> tuple[str, bool]:
    """Keep the head and tail of oversized output, elide the middle.

    Compilers report the first errors; test runners report failures at the end.
    Keeping both ends preserves nearly all diagnostic value.
    """
    data = text.encode("utf-8", "replace")
    if len(data) <= limit:
        return text, False
    head = data[: limit // 2].decode("utf-8", "replace")
    tail = data[-limit // 2 :].decode("utf-8", "replace")
    elided = len(data) - limit
    return f"{head}\n\n... [{elided} bytes elided] ...\n\n{tail}", True


def run(
    argv: Sequence[str] | str,
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 300.0,
    stdin: str | None = None,
    shell: bool = False,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    check: bool = False,
) -> ProcResult:
    """Run a command to completion under a hard timeout.

    ``shell=True`` is supported because build systems genuinely need pipelines,
    but the sandbox layer prefers argv form. On timeout the entire process group
    receives SIGTERM, then SIGKILL after a grace period.
    """
    if shell:
        if not isinstance(argv, str):
            argv = " ".join(argv)
        args: Sequence[str] | str = argv
        display = [argv]
    else:
        if isinstance(argv, str):
            raise SandboxError("argv must be a sequence when shell=False", argv=argv)
        args = list(argv)
        display = list(argv)

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            text=True,
            errors="replace",
            start_new_session=True,  # own process group -> killable as a unit
        )
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.add(proc)
    except FileNotFoundError as exc:
        raise SandboxError(f"command not found: {display[0]}", argv=display) from exc
    except OSError as exc:
        raise SandboxError(f"failed to spawn: {exc}", argv=display) from exc

    timed_out = False
    try:
        try:
            out, err = proc.communicate(input=stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            try:
                out, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - kill -9 already sent
                out, err = "", ""
    except BaseException:
        # Anything else raised while waiting -- KeyboardInterrupt, a signal
        # handler that raises (pytest-timeout does exactly this), a bug in the
        # reader -- used to propagate with the child still running and still
        # registered. That contradicts this module's first promise and was
        # observed leaving `opencode` process groups alive for twenty minutes
        # after their parent had gone.
        _kill_group(proc)
        raise
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)
    duration = time.monotonic() - start

    out, t1 = clamp_output(out or "", output_limit)
    err, t2 = clamp_output(err or "", output_limit)

    result = ProcResult(
        argv=display,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=out,
        stderr=err,
        duration=duration,
        timed_out=timed_out,
        truncated=t1 or t2,
        cwd=str(cwd) if cwd else None,
        env_overrides=dict(env or {}),
    )

    if timed_out:
        raise CommandTimeout(
            f"command exceeded {timeout:.0f}s",
            argv=display,
            tail=result.tail(20),
        )
    if check and not result.ok:
        raise SandboxError(
            f"command failed with exit {result.returncode}",
            argv=display,
            tail=result.tail(20),
        )
    return result


def _kill_group(proc: subprocess.Popen, *, grace: float = 5.0) -> None:
    """SIGTERM the process group, escalate to SIGKILL after a grace period."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):  # pragma: no cover
        return
    for sig, wait_seconds in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            return
        if wait_seconds:
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.05)


class BackgroundProcess:
    """A long-lived child (dev server, VNC bridge) with guaranteed teardown.

    Used as a context manager so a crashing gate cannot leak a listening socket
    into the next attempt -- a failure mode that otherwise produces baffling
    "port already in use" cascades hours later.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(env or {})
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._buffer: list[str] = []
        self._reader: threading.Thread | None = None

    def start(self) -> BackgroundProcess:
        full_env = dict(os.environ)
        full_env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                self.argv,
                cwd=str(self.cwd) if self.cwd else None,
                env=full_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"failed to start background process: {exc}", argv=self.argv) from exc
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        return self

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        handle = self.log_path.open("a", encoding="utf-8") if self.log_path else None
        try:
            for line in self._proc.stdout:
                self._buffer.append(line)
                if len(self._buffer) > 2000:
                    del self._buffer[:1000]
                if handle:
                    handle.write(line)
                    handle.flush()
        except (ValueError, OSError):  # pragma: no cover - stream closed on teardown
            pass
        finally:
            if handle:
                handle.close()

    @property
    def output(self) -> str:
        return "".join(self._buffer)

    def poll(self) -> int | None:
        return self._proc.poll() if self._proc else None

    def wait_for(self, predicate, timeout: float = 60.0, interval: float = 0.25) -> bool:
        """Poll ``predicate()`` until true, the process dies, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return False
            if predicate():
                return True
            time.sleep(interval)
        return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            _kill_group(self._proc)
        if self._reader:
            self._reader.join(timeout=2)

    def __enter__(self) -> BackgroundProcess:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def which(name: str) -> str | None:
    """Locate an executable without importing shutil at call sites."""
    import shutil

    return shutil.which(name)
