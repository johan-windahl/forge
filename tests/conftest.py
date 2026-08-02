"""Shared fixtures.

Every test runs against the echo provider and a manual clock, so the suite needs
no network, no models and no real time. That is deliberate: a test suite for an
autonomous platform that depended on a model would be slow, flaky, and unable to
assert anything precise about routing or retries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge.config import Config, ProviderConfig, default_models
from forge.kernel.ledger import Ledger
from forge.models.provider import EchoProvider
from forge.util.clock import ManualClock


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(project_dir=tmp_path, models=default_models())
    cfg.scheduler.workers = 1
    cfg.scheduler.poll_interval = 0.01
    cfg.scheduler.lease_seconds = 60.0
    cfg.scheduler.lease_renew_interval = 30.0
    cfg.scheduler.backoff_base = 0.01
    cfg.scheduler.backoff_max = 0.05
    cfg.validation.gates = ["schema"]
    # `auto` resolves to OpenCode whenever the binary is on PATH, and OpenCode
    # talks to the model itself rather than through the echo providers below.
    # Left at the default, coding tests drove a real local model and timed out.
    cfg.coding.backend = "native"
    cfg.memory.lessons_global_path = str(tmp_path / "lessons")
    cfg.models.cache_enabled = False
    for provider in cfg.models.providers.values():
        provider.kind = "echo"
        provider.api_key_env = ""
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def ledger(config: Config, clock: ManualClock) -> Ledger:
    instance = Ledger(config.ledger_path, clock=clock, project_id="proj_test")
    yield instance
    instance.close()


class ScriptedProvider(EchoProvider):
    """An echo provider that answers from a queue keyed by task class or order."""

    def __init__(self, name: str = "echo") -> None:
        super().__init__(name, ProviderConfig(kind="echo", base_url="", api_key_env=""))
        self.by_class: dict[str, list[str]] = {}
        self.fail_with: Exception | None = None
        self.fail_times: int = 0

    def queue(self, task_class: str, payload: Any) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.by_class.setdefault(str(task_class), []).append(text)

    def _complete(self, request, spec):  # type: ignore[no-untyped-def]
        if self.fail_times > 0 and self.fail_with is not None:
            self.fail_times -= 1
            raise self.fail_with
        queued = self.by_class.get(str(request.profile.task_class))
        if queued:
            self.responses.insert(0, queued.pop(0))
        return super()._complete(request, spec)


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider()


@pytest.fixture
def orchestrator(config: Config, provider: ScriptedProvider):
    from forge.kernel.orchestrator import Orchestrator

    instance = Orchestrator(config, worker_prefix="test")
    for name in config.models.providers:
        instance.models.registry.install(name, provider)
    yield instance
    instance.close()
