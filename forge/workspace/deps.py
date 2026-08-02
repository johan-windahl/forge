"""Installing a project's dependencies, when and only when it is needed.

A generated project is not runnable the moment its manifest is written. The
scaffold node produces ``package.json`` and ``tsconfig.json``; nothing has
fetched ``node_modules``, so every tool the validation layer wants to run is
absent. Observed live, that produced the worst failure mode in the system: the
types gate reported "``npx --no-install tsc --noEmit`` failed", the node treated
it as a code defect, discarded a perfectly good local result and escalated to
the costliest rung -- repeatedly, over a compiler that was simply not installed.

Skipping the gate stops the bleeding but gives up the checking. Installing is
the actual answer, and it belongs here rather than in a prompt, for a reason
worth stating: a model instructed to "run npm install first" will comply most of
the time, and the failure mode of *most of the time* is this same expensive
cascade in the cases it forgets. Deterministic infrastructure does not forget.

Three properties matter more than convenience:

**Idempotence.** Installing is slow and network-bound, so it must happen once
per meaningful change and not once per node. The trigger is a fingerprint of the
manifest and lockfile; identical fingerprint means the existing tree is current.

**Bounded failure.** A failing install must not become its own retry loop -- the
exact bug this module exists to fix. A fingerprint that failed is recorded and
not retried, so a broken manifest degrades to "gates skip" rather than
"everything spins".

**Reproducibility.** Where a lockfile exists, the locked installer is used
(``npm ci``, ``poetry install``, ``cargo fetch --locked``). A build that resolves
different versions on Tuesday than it did on Monday is not reproducible, and the
whole validation layer rests on runs being comparable.

Security note, stated plainly: installing dependencies executes third-party
code, including ``postinstall`` hooks, with whatever access the sandbox allows.
That is inherent to building software with a package manager, not something this
module introduces, and the sandbox is the boundary that contains it. It is also
why ``install_dependencies`` is a configuration switch rather than an
unconditional behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..obs.log import get_logger
from ..util.hashing import content_hash

log = get_logger("workspace.deps")

#: Legacy location. This was inside the repository and became tracked when a
#: scaffold replaced Forge's starter ``.gitignore``. Different worktrees then
#: committed different fingerprints and every otherwise-clean integration
#: conflicted on Forge's own metadata.
STAMP_PATH = ".forge/deps-stamp.json"

#: Installing is network-bound and occasionally enormous. Generous, but finite.
DEFAULT_TIMEOUT = 900.0


@dataclass(slots=True)
class InstallPlan:
    """How to install this project's dependencies."""

    language: str
    command: str
    #: Files whose content decides whether an install is still current.
    fingerprint_files: list[str]
    #: Directory whose absence proves nothing has been installed.
    marker: str
    #: True when the command respects a lockfile exactly.
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "command": self.command,
            "marker": self.marker,
            "locked": self.locked,
        }


def plan_for(sandbox: Any, toolchain: dict[str, Any]) -> InstallPlan | None:
    """Work out the install command from marker files, not from a model.

    Prefers the locked form wherever a lockfile is present: reproducibility is
    worth more here than the small chance that the lockfile is stale.
    """
    languages = toolchain.get("languages", [])

    if "node" in languages:
        manager = str(toolchain.get("package_manager", "npm"))
        if not _has_dependencies(sandbox):
            return None  # a manifest with no dependencies needs no install
        if manager == "pnpm":
            locked = sandbox.exists("pnpm-lock.yaml")
            command = "pnpm install --frozen-lockfile" if locked else "pnpm install"
            files = ["package.json", "pnpm-lock.yaml"]
        elif manager == "yarn":
            locked = sandbox.exists("yarn.lock")
            command = "yarn install --immutable" if locked else "yarn install"
            files = ["package.json", "yarn.lock"]
        else:
            locked = sandbox.exists("package-lock.json")
            # An unlocked install normally creates package-lock.json. In a node
            # worktree that file gets committed, while the same install leaves
            # an untracked copy in main; Git then refuses the integration
            # because the untracked main copy would be overwritten.
            command = "npm ci" if locked else "npm install --no-package-lock"
            files = ["package.json", "package-lock.json"]
        return InstallPlan("node", command, files, "node_modules", locked)

    if "python" in languages:
        if sandbox.exists("poetry.lock"):
            return InstallPlan(
                "python", "poetry install", ["pyproject.toml", "poetry.lock"], ".venv", True
            )
        if sandbox.exists("requirements.txt"):
            return InstallPlan(
                "python", "pip install -r requirements.txt", ["requirements.txt"], ".venv"
            )
        if sandbox.exists("pyproject.toml"):
            return InstallPlan("python", "pip install -e .", ["pyproject.toml"], ".venv")

    if "rust" in languages:
        locked = sandbox.exists("Cargo.lock")
        return InstallPlan(
            "rust",
            "cargo fetch --locked" if locked else "cargo fetch",
            ["Cargo.toml", "Cargo.lock"],
            "target",
            locked,
        )

    if "go" in languages:
        return InstallPlan("go", "go mod download", ["go.mod", "go.sum"], "", False)

    return None


def _has_dependencies(sandbox: Any) -> bool:
    """A package.json with no dependencies at all does not need installing."""
    try:
        package = json.loads(sandbox.read("package.json"))
    except Exception:  # pragma: no cover - malformed manifest
        return False
    return bool(package.get("dependencies") or package.get("devDependencies"))


def fingerprint(sandbox: Any, plan: InstallPlan) -> str:
    """Hash the files that decide whether an install is still current.

    Deliberately *not* keyed on the install command. The manifest content is
    the thing that decides whether the installed tree is current; how it got
    installed is not.
    """
    parts: list[str] = []
    for name in plan.fingerprint_files:
        try:
            parts.append(f"{name}:{sandbox.read(name)}")
        except Exception:
            parts.append(f"{name}:absent")
    return content_hash(*parts)


def stamp_path(plan: InstallPlan) -> str:
    """A derived location that cannot become project source."""
    marker = plan.marker.rstrip("/")
    return f"{marker}/.forge-deps-stamp.json" if marker else STAMP_PATH


def _read_stamp(sandbox: Any, plan: InstallPlan) -> dict[str, Any]:
    try:
        return dict(json.loads(sandbox.read(stamp_path(plan))))
    except Exception:
        # Read the legacy record during migration, but never write it again.
        try:
            return dict(json.loads(sandbox.read(STAMP_PATH)))
        except Exception:
            return {}


def _write_stamp(sandbox: Any, plan: InstallPlan, data: dict[str, Any]) -> None:
    try:
        sandbox.write(stamp_path(plan), json.dumps(data, indent=2))
    except Exception as exc:  # pragma: no cover - unwritable workspace
        log.warn("could not record dependency stamp", error=str(exc))


def needs_install(sandbox: Any, plan: InstallPlan) -> tuple[bool, str]:
    """Should an install run now? Returns (decision, reason).

    Answering "no" is the common case and must be cheap: it is consulted after
    every node that touches a manifest.
    """
    current = fingerprint(sandbox, plan)
    stamp = _read_stamp(sandbox, plan)

    if stamp.get("failed_fingerprint") == current:
        # Retrying an install that already failed on these exact inputs is the
        # loop this module exists to avoid.
        return False, "an install already failed for this manifest; not retrying"
    if stamp.get("fingerprint") == current and (not plan.marker or sandbox.exists(plan.marker)):
        return False, "dependencies are current"
    if plan.marker and not sandbox.exists(plan.marker):
        return True, f"{plan.marker} is missing"
    if stamp.get("fingerprint") != current:
        return True, "the manifest or lockfile changed"
    return False, "dependencies are current"


def ensure(
    sandbox: Any,
    toolchain: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    enabled: bool = True,
) -> dict[str, Any] | None:
    """Install dependencies if the project needs them. Returns a result record.

    Returns ``None`` when nothing was attempted, which is the overwhelmingly
    common case and must stay silent -- a line of log per node would bury the
    one time it mattered.
    """
    if not enabled:
        return None
    plan = plan_for(sandbox, toolchain)
    if plan is None:
        return None

    should, reason = needs_install(sandbox, plan)
    if not should:
        return None

    log.info("installing dependencies", command=plan.command, reason=reason, locked=plan.locked)
    try:
        result = sandbox.exec(plan.command, shell=True, timeout=timeout)
    except Exception as exc:
        log.warn("dependency install could not run", command=plan.command, error=str(exc))
        _write_stamp(sandbox, plan, {
            "failed_fingerprint": fingerprint(sandbox, plan),
            "command": plan.command,
            "error": str(exc),
        })
        return {**plan.to_dict(), "ok": False, "reason": reason, "error": str(exc)}

    current = fingerprint(sandbox, plan)
    if result.ok:
        _write_stamp(sandbox, plan, {"fingerprint": current, "command": plan.command})
        log.info("dependencies installed", command=plan.command, duration=round(result.duration, 1))
    else:
        # Recorded so the next node does not try the same thing again. Gates
        # will skip rather than fail, which is the correct degraded behaviour.
        _write_stamp(sandbox, plan, {
            "failed_fingerprint": current,
            "command": plan.command,
            "tail": result.tail(20),
        })
        log.warn("dependency install failed", command=plan.command, tail=result.tail(10))

    return {
        **plan.to_dict(),
        "ok": result.ok,
        "reason": reason,
        "duration": round(result.duration, 2),
        "tail": "" if result.ok else result.tail(20),
    }
