"""Dependency bootstrapping.

Written against the live incident: a scaffolded project had package.json and
tsconfig.json but no node_modules, so `npx --no-install tsc` failed, the node
read that as a code defect, discarded a good local result and escalated to a
frontier rung. Installing is the fix; the tests below hold it to the three
properties that make it safe to do automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.config import SandboxConfig
from forge.workspace.deps import ensure, needs_install, plan_for, stamp_path
from forge.workspace.sandbox import LocalSandbox, detect_toolchain


def _sandbox(tmp_path: Path) -> LocalSandbox:
    root = tmp_path / "wt"
    root.mkdir()
    (root / ".forge").mkdir()
    return LocalSandbox(SandboxConfig(command_timeout=20), root)


class _StubSandbox:
    """A sandbox whose `exec` is scripted, so no package manager really runs.

    The suite does not touch the network, and an install test that shells out to
    a real npm would be both slow and dependent on a registry being reachable.
    File operations still go to a real LocalSandbox, because the fingerprint and
    stamp logic is exactly what is under test.
    """

    def __init__(self, inner: LocalSandbox, *, ok: bool = True) -> None:
        self._inner = inner
        self.ok = ok
        self.commands: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def exec(self, command: str, **kwargs):
        self.commands.append(command)
        if self.ok:
            if "--no-package-lock" not in command:
                self._inner.write("package-lock.json", '{"lockfileVersion": 3}')
            (self._inner.root / "node_modules").mkdir(exist_ok=True)
        return self._inner.exec("true" if self.ok else "false", shell=True, timeout=20)


def _node_project(sandbox: LocalSandbox, *, lockfile: bool = False, deps: bool = True) -> dict:
    package = {"name": "p", "version": "1.0.0"}
    if deps:
        package["devDependencies"] = {"typescript": "^5.0.0"}
    sandbox.write("package.json", json.dumps(package))
    if lockfile:
        sandbox.write("package-lock.json", '{"lockfileVersion": 3}')
    return detect_toolchain(sandbox)


# --------------------------------------------------------------------------
# Choosing the command
# --------------------------------------------------------------------------


def test_a_lockfile_selects_the_reproducible_installer(tmp_path: Path) -> None:
    """`npm ci` resolves exactly what the lockfile says; `npm install` may not.

    Validation rests on runs being comparable, so a build that resolves
    different versions on Tuesday than on Monday is not acceptable.
    """
    sandbox = _sandbox(tmp_path)
    plan = plan_for(sandbox, _node_project(sandbox, lockfile=True))
    assert plan is not None and plan.command == "npm ci" and plan.locked


def test_without_a_lockfile_the_unlocked_installer_is_used(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    plan = plan_for(sandbox, _node_project(sandbox, lockfile=False))
    assert plan is not None
    assert plan.command == "npm install --no-package-lock"
    assert not plan.locked


def test_a_manifest_with_no_dependencies_needs_no_install(tmp_path: Path) -> None:
    """Running a package manager to install nothing is pure latency."""
    sandbox = _sandbox(tmp_path)
    assert plan_for(sandbox, _node_project(sandbox, deps=False)) is None


def test_a_project_with_no_manifest_has_no_plan(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    sandbox.write("index.html", "<!doctype html>")
    assert plan_for(sandbox, detect_toolchain(sandbox)) is None


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------


def test_install_is_needed_when_the_marker_is_absent(tmp_path: Path) -> None:
    sandbox = _sandbox(tmp_path)
    plan = plan_for(sandbox, _node_project(sandbox))
    should, reason = needs_install(sandbox, plan)
    assert should and "node_modules" in reason


def test_install_is_skipped_once_dependencies_are_current(tmp_path: Path) -> None:
    """Installing is slow and network-bound; it must happen per change, not per node."""
    inner = _sandbox(tmp_path)
    toolchain = _node_project(inner)
    sandbox = _StubSandbox(inner)
    ensure(sandbox, toolchain, timeout=20)  # records the stamp

    should, reason = needs_install(sandbox, plan_for(sandbox, toolchain))
    assert not should and "current" in reason


def test_a_changed_manifest_triggers_a_reinstall(tmp_path: Path) -> None:
    inner = _sandbox(tmp_path)
    toolchain = _node_project(inner)
    sandbox = _StubSandbox(inner)
    ensure(sandbox, toolchain, timeout=20)

    sandbox.write("package.json", json.dumps(
        {"name": "p", "version": "1.0.0", "devDependencies": {"typescript": "^5.0.0", "vite": "^5"}}
    ))
    should, reason = needs_install(sandbox, plan_for(sandbox, toolchain))
    assert should and "changed" in reason


# --------------------------------------------------------------------------
# Bounded failure -- the property this module exists to protect
# --------------------------------------------------------------------------


def test_a_failed_install_is_not_retried_for_the_same_manifest(tmp_path: Path) -> None:
    """A failing install must not become the retry loop it was written to prevent.

    Degrading to "gates skip" is correct; spinning on a broken manifest is the
    exact cascade that made a missing compiler cost frontier-rung escalations.
    """
    from forge.workspace.deps import fingerprint

    sandbox = _sandbox(tmp_path)
    toolchain = _node_project(sandbox)
    plan = plan_for(sandbox, toolchain)
    assert plan is not None
    sandbox.write(stamp_path(plan), json.dumps({
        "failed_fingerprint": fingerprint(sandbox, plan),
        "command": "npm ci",
    }))

    should, reason = needs_install(sandbox, plan)
    assert not should
    assert "not retrying" in reason


def test_a_failed_install_records_its_fingerprint(tmp_path: Path) -> None:
    inner = _sandbox(tmp_path)
    toolchain = _node_project(inner)
    sandbox = _StubSandbox(inner, ok=False)
    plan = plan_for(sandbox, toolchain)
    assert plan is not None

    record = ensure(sandbox, toolchain, timeout=20)
    assert record is not None and not record["ok"]
    stamp = json.loads(sandbox.read(stamp_path(plan)))
    assert stamp.get("failed_fingerprint"), "the failure must be remembered"
    assert not stamp.get("fingerprint"), "a failure is not a successful install"

    should, _ = needs_install(sandbox, plan)
    assert not should, "the same broken manifest must not be attempted twice"


def test_installing_can_be_switched_off(tmp_path: Path) -> None:
    """Installing runs third-party postinstall hooks; declining must be possible."""
    sandbox = _sandbox(tmp_path)
    assert ensure(sandbox, _node_project(sandbox), enabled=False) is None


def test_a_successful_install_is_recorded_and_silences_the_next_check(tmp_path: Path) -> None:
    inner = _sandbox(tmp_path)
    toolchain = _node_project(inner)
    sandbox = _StubSandbox(inner)

    record = ensure(sandbox, toolchain, timeout=20)
    assert record is not None and record["ok"]
    plan = plan_for(sandbox, toolchain)
    assert plan is not None
    assert json.loads(sandbox.read(stamp_path(plan))).get("fingerprint")

    # The unlocked install deliberately does not create project source.
    assert ensure(sandbox, toolchain, timeout=20) is None, "a second call must do nothing"
    assert len(sandbox.commands) == 1
