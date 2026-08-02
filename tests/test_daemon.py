"""Detached runs: the pidfile is the only thing standing between one
orchestrator and two.

These tests use real processes. A pidfile abstraction tested against a mocked
``os.kill`` proves nothing -- the whole point is behaviour against the actual
process table, including the case the naive implementation gets wrong: a PID
that has been recycled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from forge.config import Config
from forge.kernel import daemon


def _sleeper(seconds: float = 30.0) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


@pytest.fixture
def forge_dir(tmp_path: Path) -> Path:
    (tmp_path / "logs").mkdir()
    return tmp_path


def _stubborn(forge_dir: Path) -> daemon.RunHandle:
    """A detached child that ignores SIGTERM, like a node finishing its work.

    It touches a marker file once its handler is installed, and the caller waits
    for that: signalling a child that has not finished starting up kills it by
    default action, which would make this a test of interpreter startup time.
    """
    ready = forge_dir / "ready"
    handle = daemon.spawn(
        [
            sys.executable,
            "-c",
            "import signal, time, pathlib; "
            "signal.signal(signal.SIGTERM, lambda *a: None); "
            f"pathlib.Path({str(ready)!r}).write_text('x'); "
            "time.sleep(30)",
        ],
        forge_dir=forge_dir,
        cwd=forge_dir,
    )
    for _ in range(100):
        if ready.exists():
            return handle
        time.sleep(0.05)
    daemon.stop(handle, timeout=0.0, kill=True)
    pytest.fail("child never signalled readiness")


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


def test_a_live_process_is_reported_alive(forge_dir: Path) -> None:
    proc = _sleeper()
    try:
        handle = daemon.RunHandle(pid=proc.pid, identity=daemon.process_identity(proc.pid))
        assert handle.alive
    finally:
        proc.kill()
        proc.wait()


def test_a_dead_process_is_reported_dead(forge_dir: Path) -> None:
    proc = _sleeper(0.01)
    pid = proc.pid
    proc.wait()
    assert not daemon.RunHandle(pid=pid).alive


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc for process identity")
def test_a_recycled_pid_is_not_mistaken_for_the_old_run(forge_dir: Path) -> None:
    """The failure mode that makes a naive pidfile dangerous.

    PIDs wrap. Sooner or later the number in the pidfile belongs to somebody
    else's process, and a run that answers `os.kill(pid, 0)` looks alive when it
    is long gone -- so `forge run` refuses to start, forever, until someone
    deletes the file by hand.
    """
    proc = _sleeper()
    try:
        # Same PID, an identity token from a different process: this is exactly
        # what recycling looks like from the outside.
        stale = daemon.RunHandle(pid=proc.pid, identity="0000000000")
        assert not stale.alive
        assert daemon.RunHandle(pid=proc.pid, identity=daemon.process_identity(proc.pid)).alive
    finally:
        proc.kill()
        proc.wait()


def test_an_unknowable_identity_counts_as_alive(forge_dir: Path) -> None:
    """Refusing to start is recoverable; double-running is not.

    When the platform cannot supply an identity token, the ambiguous case has to
    resolve towards *not* starting a second orchestrator.
    """
    proc = _sleeper()
    try:
        assert daemon.RunHandle(pid=proc.pid, identity="").alive
    finally:
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------
# Claiming
# --------------------------------------------------------------------------


def test_a_second_claim_is_refused_while_the_first_lives(forge_dir: Path) -> None:
    daemon.claim(forge_dir)  # claimed by this very process, which is alive
    with pytest.raises(daemon.AlreadyRunning):
        daemon.claim(forge_dir)


def test_a_stale_claim_does_not_block_a_new_run(forge_dir: Path) -> None:
    """A power cut mid-run must not need a manual `rm` before building again."""
    proc = _sleeper(0.01)
    proc.wait()
    daemon.write_handle(forge_dir, daemon.RunHandle(pid=proc.pid, token="old"))

    token = daemon.claim(forge_dir)
    assert daemon.read_handle(forge_dir).token == token


def test_the_child_adopts_the_claim_made_for_it(forge_dir: Path) -> None:
    """Otherwise the spawned run finds a live pidfile -- its own -- and refuses.

    The parent has to claim before it knows the child's PID, so adoption is by
    matching secret rather than by process identity.
    """
    token = daemon.claim(forge_dir)
    adopted = daemon.claim(forge_dir, token=token)
    assert adopted == token
    assert daemon.read_handle(forge_dir).pid == os.getpid()


def test_adoption_keeps_the_log_path_the_parent_recorded(forge_dir: Path) -> None:
    """`spawn` writes where the output goes; the child overwrote it with blanks.

    `mark_running` preserves whatever it finds, so once adoption dropped `log`
    the path was gone for good: `forge status` printed no log line, and the
    operator had no way to find the output of their own multi-day run.
    """
    daemon.write_handle(
        forge_dir,
        daemon.RunHandle(pid=os.getpid(), token="t", log="/var/log/forge/run.log", cwd="/srv/p"),
    )

    daemon.claim(forge_dir, token="t")

    handle = daemon.read_handle(forge_dir)
    assert handle.log == "/var/log/forge/run.log"
    assert handle.cwd == "/srv/p"


def test_a_stale_pidfile_is_only_cleared_by_the_starter_that_read_it(
    forge_dir: Path,
) -> None:
    """Two starters racing over one stale pidfile both saw the dead handle.

    A cleared it and won the exclusive create; B, still holding its own read of
    the *dead* handle, then deleted A's fresh pidfile and won a create of its
    own. Two orchestrators, one ledger, one workspace. The checked form is what
    `release` has always done, for exactly this reason.
    """
    proc = _sleeper(0.01)
    proc.wait()
    stale = daemon.RunHandle(pid=proc.pid, token="dead")
    daemon.write_handle(forge_dir, stale)

    # A wins the race and replaces the pidfile.
    daemon.write_handle(forge_dir, daemon.RunHandle(pid=os.getpid(), token="winner"))

    # B tries to clear what it read a moment ago.
    assert daemon.clear_if(forge_dir, stale) is False
    assert daemon.read_handle(forge_dir).token == "winner"


def test_release_leaves_a_successors_claim_alone(forge_dir: Path) -> None:
    """A killed run still exiting must not delete the pidfile of its replacement."""
    daemon.write_handle(forge_dir, daemon.RunHandle(pid=os.getpid(), token="new-run"))
    assert daemon.release(forge_dir, "old-run") is False
    assert daemon.read_handle(forge_dir) is not None
    assert daemon.release(forge_dir, "new-run") is True
    assert daemon.read_handle(forge_dir) is None


def test_a_corrupt_pidfile_is_ignored_rather_than_fatal(forge_dir: Path) -> None:
    daemon.pidfile_path(forge_dir).write_text("{ truncated", encoding="utf-8")
    assert daemon.read_handle(forge_dir) is None
    assert daemon.active_run(forge_dir) is None
    daemon.claim(forge_dir)  # and the project is still startable


def test_active_run_clears_a_dead_pidfile(forge_dir: Path) -> None:
    proc = _sleeper(0.01)
    proc.wait()
    daemon.write_handle(forge_dir, daemon.RunHandle(pid=proc.pid, token="t"))
    assert daemon.active_run(forge_dir) is None
    assert not daemon.pidfile_path(forge_dir).exists()


# --------------------------------------------------------------------------
# Spawning and stopping
# --------------------------------------------------------------------------


def test_a_spawned_run_outlives_its_own_process_group(forge_dir: Path) -> None:
    """The nohup property: signals aimed at the starting shell must not reach it."""
    handle = daemon.spawn(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        forge_dir=forge_dir,
        cwd=forge_dir,
        token="t",
    )
    try:
        assert handle.alive
        assert os.getpgid(handle.pid) != os.getpgid(os.getpid())
        assert daemon.read_handle(forge_dir).pid == handle.pid
    finally:
        daemon.stop(handle, timeout=0.0, kill=True)


def test_spawn_records_output_to_the_run_log(forge_dir: Path) -> None:
    handle = daemon.spawn(
        [sys.executable, "-c", "print('hello from the run')"],
        forge_dir=forge_dir,
        cwd=forge_dir,
    )
    for _ in range(50):
        if not handle.alive:
            break
        time.sleep(0.05)
    text = Path(handle.log).read_text(encoding="utf-8")
    assert "hello from the run" in text
    assert "forge run detached" in text  # the header, so a log of many runs is readable


def test_stop_terminates_a_run(forge_dir: Path) -> None:
    handle = daemon.spawn(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        forge_dir=forge_dir,
        cwd=forge_dir,
    )
    assert daemon.stop(handle, timeout=10.0) == "stopped"
    assert not handle.alive


def test_stop_reports_winding_down_rather_than_killing(forge_dir: Path) -> None:
    """A node can hold a 30-minute lease. Reporting that is honest; SIGKILL is not.

    The child here ignores SIGTERM, which is what an orchestrator finishing an
    in-flight node looks like from outside.
    """
    handle = _stubborn(forge_dir)
    try:
        assert daemon.stop(handle, timeout=1.0) == "winding-down"
        assert handle.alive
    finally:
        daemon.stop(handle, timeout=0.0, kill=True)


def test_kill_forces_a_run_that_will_not_leave(forge_dir: Path) -> None:
    handle = _stubborn(forge_dir)
    assert daemon.stop(handle, timeout=0.5, kill=True) == "killed"
    assert not handle.alive


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc process tree")
def test_kill_also_terminates_children_in_separate_process_groups(
    forge_dir: Path,
) -> None:
    child_pid_file = forge_dir / "child.pid"
    script = (
        "import pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *a: None); "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, lambda *a: None); time.sleep(30)'], "
        "start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    handle = daemon.spawn(
        [sys.executable, "-c", script],
        forge_dir=forge_dir,
        cwd=forge_dir,
    )
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.05)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text())

    assert child_pid in daemon._descendant_pids(handle.pid)
    assert daemon.stop(handle, timeout=0.0, kill=True) == "killed"
    for _ in range(100):
        stat = Path(f"/proc/{child_pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("separate child process group survived forced stop")


def test_stopping_a_run_that_already_exited_is_not_an_error(forge_dir: Path) -> None:
    proc = _sleeper(0.01)
    proc.wait()
    assert daemon.stop(daemon.RunHandle(pid=proc.pid), timeout=0.0) == "gone"


def test_the_run_log_is_rotated_rather_than_growing_without_bound(forge_dir: Path) -> None:
    log = daemon.log_path(forge_dir)
    log.write_text("x" * 4096, encoding="utf-8")
    daemon.rotate_log(log, max_bytes=1024)
    assert log.with_suffix(log.suffix + ".1").exists()
    assert not log.exists()


# --------------------------------------------------------------------------
# The CLI contract
# --------------------------------------------------------------------------


def test_run_detaches_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: `forge run` returns, and the build keeps going."""
    from forge import cli

    calls: list[list[str]] = []

    def fake_spawn(argv, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(argv)
        return daemon.RunHandle(pid=os.getpid(), log=str(tmp_path / "logs" / "run.log"))

    monkeypatch.setattr(daemon, "spawn", fake_spawn)
    monkeypatch.setattr(daemon, "await_start", lambda handle, **kw: True)
    (tmp_path / ".forge").mkdir()
    (tmp_path / ".forge" / "ledger.db").write_text("", encoding="utf-8")

    assert cli.main(["--dir", str(tmp_path), "run", "--forever"]) == 0
    assert len(calls) == 1
    # The child runs the same command in the foreground, so a flag added to
    # `forge run` later is passed through without touching the daemon code.
    assert calls[0][-3:] == ["run", "--forever", "--foreground"]


def test_run_refuses_to_start_a_second_orchestrator(tmp_path: Path) -> None:
    """Two runs on one ledger would race on node leases, git and the workspace."""
    from forge import cli

    (tmp_path / ".forge").mkdir()
    daemon.write_handle(
        tmp_path / ".forge",
        daemon.RunHandle(pid=os.getpid(), identity=daemon.process_identity(os.getpid()), token="t"),
    )
    # Exit 0, not an error: the operator asked for a run to be active and one is.
    assert cli.main(["--dir", str(tmp_path), "run"]) == 0


def test_foreground_still_runs_in_this_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from forge import cli

    called: list[bool] = []
    monkeypatch.setattr(cli, "_run_here", lambda config, args: called.append(True) or 0)
    assert cli.main(["--dir", str(tmp_path), "run", "--foreground"]) == 0
    assert called == [True]


def test_a_foreground_run_also_claims_the_pidfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise `forge status` calls a live foreground run 'crashed'."""
    from forge import cli

    seen: list[str] = []

    def observe(config: Config, args: object) -> int:
        handle = daemon.active_run(config.forge_dir)
        seen.append(handle.state if hasattr(handle, "state") else str(handle.pid))
        return 0

    monkeypatch.setattr(cli, "_run_here", observe)
    cli.main(["--dir", str(tmp_path), "run", "--foreground"])
    assert seen == [str(os.getpid())]
    # And it is released on the way out, so the next run is not blocked.
    assert daemon.active_run(tmp_path / ".forge") is None


def test_stop_says_so_when_nothing_is_running(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from forge import cli

    (tmp_path / ".forge").mkdir()
    assert cli.main(["--dir", str(tmp_path), "stop"]) == 0
    assert "No run is active" in capsys.readouterr().out


def test_the_claim_variable_is_not_mistaken_for_configuration(tmp_path: Path) -> None:
    """`FORGE_*` is the config namespace, and unknown keys in it are an error.

    Caught the first time a detached run was started for real: the child inherited
    FORGE_RUN_CLAIM, config loading rejected it as an unknown setting, and every
    run died in its first second.
    """
    from forge.config import load_config

    load_config(tmp_path, environ={"FORGE_RUN_CLAIM": "abc123"})  # must not raise
