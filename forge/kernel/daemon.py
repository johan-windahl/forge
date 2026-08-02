"""Detached run supervision.

A Forge run lasts hours or days. Tying it to a terminal session is wrong by
default: the operator closes a laptop, the SSH connection drops, and a build
that was thirty nodes deep dies with the shell. So ``forge run`` puts the run in
its own session, redirects its output to a file, and returns immediately.

This module is the bookkeeping that makes that safe. Three problems have to be
solved before backgrounding is an improvement rather than a way to lose work:

1. **Only one run at a time.** Two orchestrators on one ledger would race on
   node leases, git resets and the workspace. The pidfile is claimed with
   ``O_CREAT|O_EXCL`` so two simultaneous ``forge run`` invocations cannot both
   win.
2. **A dead run must not look alive.** PIDs are recycled. A pidfile alone will
   eventually name somebody else's process, and a stale one makes ``forge run``
   refuse to start forever. Liveness therefore checks process *identity* -- on
   Linux, the kernel's start-time counter -- not just that some process answers
   to the number.
3. **Stopping has to be graceful.** The orchestrator winds down on SIGTERM:
   in-flight nodes finish, a checkpoint is written, ``run.stopped`` is recorded.
   SIGKILL skips all of that, so it is never the default -- it is what ``--kill``
   asks for explicitly, after the operator has been told what it costs.

Nothing here imports the orchestrator. Supervision knows about processes and
files; it does not know what the process does.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ForgeError

#: Name of the pidfile inside the state directory.
PIDFILE = "run.pid"

#: Console output of a detached run. The structured JSONL log is separate and
#: keeps its own path; this is the human stream -- what Ctrl-C used to show.
LOG_NAME = "run.log"

#: One rotation, at a size where the file is still greppable. A multi-day run
#: with retry storms can produce a lot of console output, and an unbounded log
#: in the state directory is a slow-motion disk-full incident.
LOG_MAX_BYTES = 32 * 1024 * 1024

#: How long ``stop`` waits for a graceful wind-down before reporting back. Not a
#: deadline for the run: a node holding a 30-minute lease legitimately takes
#: longer, and reporting "still winding down" is more honest than killing it.
STOP_TIMEOUT = 60.0

#: Grace period between SIGTERM and SIGKILL when ``--kill`` was asked for.
KILL_GRACE = 5.0


class AlreadyRunning(ForgeError):
    """A live run already owns this project."""

    def __init__(self, handle: RunHandle) -> None:
        super().__init__(f"a run is already active (pid {handle.pid})")
        self.handle = handle


@dataclass(slots=True)
class RunHandle:
    """Identity of a run, as recorded in the pidfile."""

    pid: int
    started_at: float = 0.0
    argv: list[str] = field(default_factory=list)
    cwd: str = ""
    log: str = ""
    #: Opaque per-process token used to detect PID reuse. Empty when the
    #: platform cannot supply one, in which case liveness degrades to the
    #: PID-only check rather than failing.
    identity: str = ""
    #: Claim token. The process that owns the claim carries this, which is how a
    #: spawned child adopts the claim its parent made for it, and how a process
    #: knows on exit whether the pidfile is still its own to remove.
    token: str = ""
    #: True while the claim is held but no orchestrator is running behind it yet.
    starting: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "started_at": self.started_at,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "log": self.log,
            "identity": self.identity,
            "token": self.token,
            "starting": self.starting,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunHandle:
        return cls(
            pid=int(data.get("pid", 0)),
            started_at=float(data.get("started_at", 0.0)),
            argv=list(data.get("argv") or []),
            cwd=str(data.get("cwd", "")),
            log=str(data.get("log", "")),
            identity=str(data.get("identity", "")),
            token=str(data.get("token", "")),
            starting=bool(data.get("starting", False)),
        )

    @property
    def alive(self) -> bool:
        """Whether *this* process -- not merely this PID -- is still running."""
        if self.pid <= 0 or not _pid_exists(self.pid):
            return False
        if _is_zombie(self.pid):
            return False
        if not self.identity:
            return True
        current = process_identity(self.pid)
        # An unreadable identity on a PID that exists means the process is there
        # but not inspectable (another user, a hardened /proc). Treat it as
        # alive: refusing to start is recoverable, double-running is not.
        return not current or current == self.identity

    def age(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.started_at)


# --------------------------------------------------------------------------
# Process identity
# --------------------------------------------------------------------------


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def _proc_stat(pid: int) -> list[str]:
    """Fields of ``/proc/<pid>/stat`` from the state field onwards.

    The second field is the executable name in parentheses and may itself contain
    spaces or parentheses, so the split has to start after the last ``)``.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    close = raw.rfind(")")
    if close < 0:
        return []
    return raw[close + 1 :].split()


def process_identity(pid: int) -> str:
    """A token that changes when a PID is reused.

    Linux exposes the process start time in jiffies as field 22 of
    ``/proc/<pid>/stat``; combined with the PID it is unique for the lifetime of
    the boot. Returns an empty string where that is unavailable, which callers
    must treat as "cannot tell" rather than "not running".
    """
    fields = _proc_stat(pid)
    # fields[0] is the state (field 3), so start time (field 22) is at index 19.
    return fields[19] if len(fields) > 19 else ""


def _is_zombie(pid: int) -> bool:
    """A process that has exited but has not been reaped by its parent.

    It still answers ``kill(pid, 0)`` and still has a ``/proc`` entry with the
    same start time, so every identity check above passes -- and the run looks
    alive forever. This happens for real whenever the process that spawned the
    run is still around, which is exactly the ``forge run --follow`` case.
    """
    fields = _proc_stat(pid)
    return bool(fields) and fields[0] == "Z"


# --------------------------------------------------------------------------
# Pidfile
# --------------------------------------------------------------------------


def pidfile_path(forge_dir: Path) -> Path:
    return Path(forge_dir) / PIDFILE


def read_handle(forge_dir: Path) -> RunHandle | None:
    """The recorded handle, alive or not. ``None`` if there is no usable file."""
    try:
        data = json.loads(pidfile_path(forge_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    handle = RunHandle.from_dict(data)
    return handle if handle.pid > 0 else None


def active_run(forge_dir: Path) -> RunHandle | None:
    """The live run for this project, clearing the pidfile if it has died.

    Self-healing on purpose: a machine that lost power mid-run must not need a
    manual ``rm`` before it can build again.
    """
    handle = read_handle(forge_dir)
    if handle is None:
        return None
    if handle.alive:
        return handle
    release(forge_dir, handle.token)
    return None


def write_handle(forge_dir: Path, handle: RunHandle) -> None:
    path = pidfile_path(forge_dir)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(handle.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def clear(forge_dir: Path) -> None:
    pidfile_path(forge_dir).unlink(missing_ok=True)


def clear_if(forge_dir: Path, expected: RunHandle) -> bool:
    """Remove the pidfile only if it still holds ``expected``.

    The unconditional form is a race when two starters find the same stale
    pidfile: A clears it and wins the exclusive create, then B -- still holding
    its own read of the dead handle -- deletes A's *fresh* pidfile and wins a
    create of its own. Two orchestrators, one ledger, one workspace.
    """
    current = read_handle(forge_dir)
    if current is None:
        return True
    if (current.pid, current.started_at, current.token) != (
        expected.pid,
        expected.started_at,
        expected.token,
    ):
        return False
    pidfile_path(forge_dir).unlink(missing_ok=True)
    return True


def claim(forge_dir: Path, *, token: str = "", argv: list[str] | None = None) -> str:
    """Take exclusive ownership of the pidfile, or raise ``AlreadyRunning``.

    The exclusive create is what makes two concurrent ``forge run`` invocations
    resolve to one winner. The placeholder left behind carries this process's own
    PID, so a crash between claiming and spawning is seen as a dead run by the
    next invocation rather than blocking it forever.

    ``token`` lets a spawned child adopt the claim its parent already made on its
    behalf -- otherwise the child would find a live pidfile (its own) and refuse
    to start. Adoption is by matching secret, not by PID, because the parent
    writes the claim before it knows the child's PID.

    Returns the claim token, which the owner passes to ``release``.
    """
    path = Path(forge_dir) / PIDFILE
    token = token or os.urandom(8).hex()
    mine = RunHandle(
        pid=os.getpid(),
        started_at=time.time(),
        identity=process_identity(os.getpid()),
        argv=list(argv or []),
        token=token,
        starting=True,
    )
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = read_handle(forge_dir)
            if existing is not None and existing.token and existing.token == token:
                # Adopt, keeping what the parent recorded. Overwriting with a
                # freshly built handle dropped `log` and `cwd`, and since
                # `mark_running` preserves whatever it finds, the only durable
                # record of where a detached run writes its output was gone:
                # `forge status` then showed no log line at all.
                mine.log = mine.log or existing.log
                mine.cwd = mine.cwd or existing.cwd
                mine.argv = mine.argv or list(existing.argv)
                write_handle(forge_dir, mine)
                return token
            if existing is not None and existing.alive:
                raise AlreadyRunning(existing) from None
            if existing is None:
                clear(forge_dir)  # unreadable: nobody's claim to protect
            else:
                # If this fails, another starter cleared the stale file and
                # claimed first. Their pidfile is not ours to delete; the retry
                # below will see a live run and defer to it.
                clear_if(forge_dir, existing)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(mine.to_dict(), indent=2))
        return token
    existing = read_handle(forge_dir)
    if existing is not None:
        raise AlreadyRunning(existing)
    return token


def release(forge_dir: Path, token: str) -> bool:
    """Remove the pidfile if it is still ours.

    Checked rather than unconditional: a run that overran its claim -- killed,
    then restarted while the old process was still exiting -- must not have its
    successor's pidfile deleted by the corpse.
    """
    handle = read_handle(forge_dir)
    if handle is not None and token and handle.token != token:
        return False
    clear(forge_dir)
    return True


def mark_running(forge_dir: Path, token: str, *, argv: list[str] | None = None) -> RunHandle | None:
    """Record that the orchestrator is now up behind an existing claim."""
    handle = read_handle(forge_dir)
    if handle is None or (token and handle.token != token):
        return None
    handle.pid = os.getpid()
    handle.identity = process_identity(os.getpid())
    handle.starting = False
    if argv is not None:
        handle.argv = list(argv)
    write_handle(forge_dir, handle)
    return handle


# --------------------------------------------------------------------------
# Spawning
# --------------------------------------------------------------------------


def log_path(forge_dir: Path) -> Path:
    return Path(forge_dir) / "logs" / LOG_NAME


def rotate_log(path: Path, *, max_bytes: int = LOG_MAX_BYTES) -> None:
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    os.replace(path, path.with_suffix(path.suffix + ".1"))


def spawn(
    argv: list[str],
    *,
    forge_dir: Path,
    cwd: Path,
    token: str = "",
    env: dict[str, str] | None = None,
) -> RunHandle:
    """Start ``argv`` detached, with output appended to the run log.

    ``start_new_session`` is the important part: a new session has no
    controlling terminal, so closing the one that started the run does not
    deliver SIGHUP to it. That is what ``nohup`` buys, done properly, and it
    also means signals sent to the shell's foreground group stop at the
    boundary rather than reaching the orchestrator.

    stdin is ``/dev/null`` rather than inherited. A model CLI that decides to
    prompt for input gets EOF and fails fast, instead of blocking forever on a
    terminal nobody is watching.
    """
    forge_dir = Path(forge_dir)
    log = log_path(forge_dir)
    log.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log)

    with open(log, "a", encoding="utf-8") as sink:
        sink.write(
            f"\n=== forge run detached at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}: "
            f"{' '.join(argv)} ===\n"
        )
        sink.flush()
        with open(os.devnull, "rb") as devnull:
            # argv is built by us and never goes through a shell.
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=devnull,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )

    handle = RunHandle(
        pid=proc.pid,
        started_at=time.time(),
        argv=list(argv),
        cwd=str(cwd),
        log=str(log),
        identity=process_identity(proc.pid),
        token=token,
        # The child clears this once its orchestrator is up; until then the
        # pidfile says "claimed but not yet running", which is the truth.
        starting=True,
    )
    # Only if the child has not already announced itself: a fast child can call
    # ``mark_running`` before we get here, and re-writing would put it back to
    # "starting" for the rest of the run.
    existing = read_handle(forge_dir)
    if existing is None or existing.starting or existing.pid != proc.pid:
        write_handle(forge_dir, handle)
    return handle


def await_start(handle: RunHandle, *, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """Wait briefly to see whether a freshly spawned run stays up.

    A misconfigured project fails in the first second -- bad config, no project,
    an import error. Without this the operator gets "started, pid 1234" for a
    process that is already gone, and only finds out on the next command.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _reap(handle.pid)
        if not handle.alive:
            return False
        time.sleep(interval)
    return handle.alive


# --------------------------------------------------------------------------
# Stopping
# --------------------------------------------------------------------------


def _reap(pid: int) -> None:
    """Clear the exit status if this process is our own child.

    Only meaningful for the ``--follow`` case, where the process that spawned the
    run is still alive and would otherwise leave a zombie behind.
    """
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def _signal_group(pid: int, sig: int) -> bool:
    """Signal the whole process group, falling back to the single process.

    Group delivery matters because a run has children: sandboxed commands, dev
    servers, model CLIs. Terminating only the orchestrator would leave them
    holding ports and CPU.
    """
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, PermissionError):
        pass
    except OSError:
        pass
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


def _descendant_pids(root_pid: int) -> set[int]:
    """Return the live Linux descendants of a process from ``/proc``.

    Model CLIs and sandbox commands deliberately create their own process
    groups, so signalling only the daemon's group cannot reach them.  Reading
    the parent links before killing the daemon preserves the tree while it is
    still knowable; after the daemon exits those children are reparented to PID
    1 and become indistinguishable from unrelated services.
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    children: dict[int, list[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
            ppid_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
            ppid = int(ppid_line.split()[1])
            pid = int(entry.name)
        except (OSError, StopIteration, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(pid)

    found: set[int] = set()
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children.get(pid, []))
    return found


def _kill_descendant_groups(root_pid: int) -> None:
    """SIGKILL every separate process group below ``root_pid``."""
    descendants = _descendant_pids(root_pid)
    if not descendants:
        return
    try:
        root_group = os.getpgid(root_pid)
    except OSError:
        root_group = root_pid
    groups: set[int] = set()
    for pid in descendants:
        try:
            group = os.getpgid(pid)
        except OSError:
            continue
        if group != root_group and group != os.getpgrp():
            groups.add(group)
    for group in groups:
        with contextlib.suppress(OSError):
            os.killpg(group, signal.SIGKILL)
    # A descendant can share neither a discoverable group leader nor the root
    # group during a concurrent fork/exit. The per-PID fallback closes that gap.
    for pid in descendants:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def stop(
    handle: RunHandle,
    *,
    timeout: float = STOP_TIMEOUT,
    kill: bool = False,
    interval: float = 0.25,
) -> str:
    """Ask a run to wind down. Returns the outcome as a short word.

    ``"gone"`` it was not running, ``"stopped"`` it exited within the timeout,
    ``"winding-down"`` SIGTERM was accepted but in-flight work is still
    finishing, ``"killed"`` it had to be forced.
    """
    if not handle.alive:
        return "gone"
    if not _signal_group(handle.pid, signal.SIGTERM):
        return "gone"

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        _reap(handle.pid)
        if not handle.alive:
            return "stopped"
        time.sleep(interval)

    if not kill:
        return "winding-down"

    _kill_descendant_groups(handle.pid)
    _signal_group(handle.pid, signal.SIGKILL)
    deadline = time.monotonic() + KILL_GRACE
    while time.monotonic() < deadline:
        _reap(handle.pid)
        if not handle.alive:
            return "killed"
        time.sleep(interval)
    return "winding-down"
