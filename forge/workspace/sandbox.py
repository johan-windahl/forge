"""Command execution, isolated to a degree the operator chooses.

Two implementations behind one interface:

``LocalSandbox``
    Runs directly on the host with a denylist and a timeout. Fast, simple, and
    appropriate when the platform builds software the operator would have built
    on that box anyway.

``DockerSandbox``
    Runs in a container with CPU, memory and optionally network limits. The
    right default for unattended multi-day operation, and the only sane choice
    if project code is ever fetched from the internet.

The interface is deliberately thin -- ``exec``, ``read``, ``write``, ``exists``
-- because every capability added here is a capability that must be secured
twice. Anything more elaborate is composed from these by the caller.

A note on the denylist: it is not a security boundary and is not presented as
one. Its job is catching the mundane accident -- a generated build script with
``rm -rf /`` in it -- before it costs an afternoon. Actual isolation is the
container's job.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import SandboxConfig
from ..errors import NotSupported, SandboxError
from ..obs.log import get_logger
from ..util.proc import DEFAULT_OUTPUT_LIMIT, BackgroundProcess, ProcResult, run, which

log = get_logger("workspace.sandbox")


class Sandbox(ABC):
    """Where commands run."""

    kind = "abstract"

    def __init__(self, config: SandboxConfig, root: Path) -> None:
        self.config = config
        self.root = Path(root).resolve()

    # -- guardrails ------------------------------------------------------

    def _check(self, argv: Sequence[str] | str) -> None:
        rendered = argv if isinstance(argv, str) else " ".join(shlex.quote(a) for a in argv)
        lowered = rendered.lower()
        for pattern in self.config.denied_prefixes:
            if pattern.lower() in lowered:
                raise SandboxError("command matched the denylist", pattern=pattern, command=rendered[:200])

    @abstractmethod
    def exec(
        self,
        argv: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        shell: bool = False,
    ) -> ProcResult: ...

    @abstractmethod
    def background(
        self, argv: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None, log_path: Path | None = None
    ) -> BackgroundProcess: ...

    # -- file access -----------------------------------------------------

    def path_for(self, relative: str = "") -> Path:
        target = (self.root / relative).resolve() if relative else self.root
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise SandboxError("path escapes the sandbox root", path=relative) from exc
        return target

    def read(self, relative: str, *, max_bytes: int = 1_000_000) -> str:
        path = self.path_for(relative)
        if not path.is_file():
            raise SandboxError("file not found", path=relative)
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", "replace")

    def write(self, relative: str, content: str) -> None:
        path = self.path_for(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def exists(self, relative: str) -> bool:
        return self.path_for(relative).exists()

    def listdir(self, relative: str = "", *, limit: int = 500) -> list[str]:
        base = self.path_for(relative)
        if not base.is_dir():
            return []
        entries = []
        for item in sorted(base.iterdir()):
            entries.append(item.name + ("/" if item.is_dir() else ""))
            if len(entries) >= limit:
                break
        return entries

    def healthy(self) -> bool:
        return True

    def teardown(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "root": str(self.root)}


class LocalSandbox(Sandbox):
    kind = "local"

    def exec(
        self,
        argv: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        shell: bool = False,
    ) -> ProcResult:
        self._check(argv)
        working = self.path_for(cwd) if cwd else self.root
        return run(
            argv,
            cwd=working,
            env=env,
            timeout=timeout or self.config.command_timeout,
            output_limit=output_limit,
            shell=shell,
        )

    def background(
        self, argv: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None, log_path: Path | None = None
    ) -> BackgroundProcess:
        self._check(argv)
        return BackgroundProcess(
            argv, cwd=self.path_for(cwd) if cwd else self.root, env=env, log_path=log_path
        ).start()


class DockerSandbox(Sandbox):
    """Runs each command in a fresh container sharing one persistent volume.

    A long-lived container would accumulate state across days of unattended
    operation -- installed packages, stray processes, edited system files -- and
    quietly become irreproducible. One container per command is slower by a
    couple of hundred milliseconds and worth every one of them, because the
    environment on day forty is the environment on day one.

    Background processes (dev servers) do get a persistent container, since they
    must outlive a single command; those are named and reaped explicitly.
    """

    kind = "docker"

    def __init__(self, config: SandboxConfig, root: Path) -> None:
        super().__init__(config, root)
        if which("docker") is None:
            raise NotSupported("docker sandbox requested but docker is not installed")
        self._containers: list[str] = []

    def _base_argv(self, *, cwd: str | None, env: Mapping[str, str] | None, detach: bool = False, name: str | None = None) -> list[str]:
        workdir = f"/work/{cwd}" if cwd else "/work"
        argv = [
            "docker",
            "run",
            "--rm",
            "-i" if not detach else "-d",
            "-v",
            f"{self.root}:/work",
            "-w",
            workdir,
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory,
            # Containers get no extra privileges and cannot escalate.
            "--security-opt",
            "no-new-privileges",
        ]
        if not self.config.allow_network:
            argv += ["--network", "none"]
        elif self.config.network:
            argv += ["--network", self.config.network]
        if name:
            argv += ["--name", name]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv.append(self.config.image)
        return argv

    def exec(
        self,
        argv: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        shell: bool = False,
    ) -> ProcResult:
        self._check(argv)
        command = argv if isinstance(argv, str) else " ".join(shlex.quote(a) for a in argv)
        full = [*self._base_argv(cwd=cwd, env=env), "/bin/sh", "-lc", command]
        result = run(
            full,
            timeout=timeout or self.config.command_timeout,
            output_limit=output_limit,
        )
        # Present the command the caller asked for, not the docker wrapper, so
        # logs and model-visible output stay about the project.
        result.argv = [command]
        return result

    def background(
        self, argv: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None, log_path: Path | None = None
    ) -> BackgroundProcess:
        self._check(argv)
        command = " ".join(shlex.quote(a) for a in argv)
        full = [*self._base_argv(cwd=cwd, env=env), "/bin/sh", "-lc", command]
        return BackgroundProcess(full, log_path=log_path).start()

    def healthy(self) -> bool:
        return run(["docker", "info"], timeout=20, check=False).ok

    def teardown(self) -> None:
        for name in self._containers:
            run(["docker", "rm", "-f", name], timeout=30, check=False)
        self._containers.clear()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "image": self.config.image, "network": self.config.network}


def build_sandbox(config: SandboxConfig, root: Path) -> Sandbox:
    match config.kind:
        case "local":
            return LocalSandbox(config, root)
        case "docker":
            return DockerSandbox(config, root)
        case other:
            raise NotSupported(f"unknown sandbox kind {other!r}")


def detect_toolchain(sandbox: Sandbox) -> dict[str, Any]:
    """Work out what the project is, deterministically.

    Reading marker files beats asking a model "what kind of project is this?" on
    every dimension that matters: it is free, instant, and cannot hallucinate a
    build command. The result seeds the gate configuration, so a Node project
    gets ``npm run build`` and a Python one gets ``pytest`` without a token
    being spent.
    """
    markers = {
        "package.json": "node",
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "Cargo.toml": "rust",
        "go.mod": "go",
        "pom.xml": "java",
        "build.gradle": "java",
        "Gemfile": "ruby",
        "composer.json": "php",
        "CMakeLists.txt": "cpp",
        "Makefile": "make",
        "index.html": "static",
    }
    found: dict[str, Any] = {"languages": [], "markers": [], "commands": {}}
    for marker, language in markers.items():
        if sandbox.exists(marker):
            found["markers"].append(marker)
            if language not in found["languages"]:
                found["languages"].append(language)

    if "node" in found["languages"]:
        try:
            import json

            package = json.loads(sandbox.read("package.json"))
            scripts = package.get("scripts", {})
            found["scripts"] = list(scripts)
            manager = "npm"
            if sandbox.exists("pnpm-lock.yaml"):
                manager = "pnpm"
            elif sandbox.exists("yarn.lock"):
                manager = "yarn"
            found["package_manager"] = manager
            for gate, script in (("build", "build"), ("unit", "test"), ("lint", "lint"), ("types", "typecheck")):
                if script in scripts:
                    found["commands"][gate] = f"{manager} run {script}"
            if "dev" in scripts:
                found["commands"]["serve"] = f"{manager} run dev"
            elif "start" in scripts:
                found["commands"]["serve"] = f"{manager} run start"
        except Exception as exc:  # pragma: no cover - malformed package.json
            log.warn("could not parse package.json", error=str(exc))

    if "python" in found["languages"]:
        found["commands"].setdefault("unit", "python -m pytest -q")
        found["commands"].setdefault("lint", "ruff check .")
        found["commands"].setdefault("types", "mypy .")
    if "rust" in found["languages"]:
        found["commands"].setdefault("build", "cargo build")
        found["commands"].setdefault("unit", "cargo test")
        found["commands"].setdefault("lint", "cargo clippy -- -D warnings")
    if "go" in found["languages"]:
        found["commands"].setdefault("build", "go build ./...")
        found["commands"].setdefault("unit", "go test ./...")
    if "static" in found["languages"] and "serve" not in found["commands"]:
        found["commands"]["serve"] = "python3 -m http.server 8000"

    return found
