"""Model registry: names to specs to live providers."""

from __future__ import annotations

import threading

from ..config import ModelsConfig, ModelSpec, ProviderConfig
from ..errors import ConfigError
from ..obs.log import get_logger
from ..util.clock import Clock
from .provider import Provider, build_provider

log = get_logger("models.registry")


class Registry:
    """Owns provider instances and answers "what can serve this request?".

    Providers are built lazily and memoised: constructing an Anthropic provider
    reads an API key, and a project that never escalates should not require one
    to be present.
    """

    def __init__(self, config: ModelsConfig, clock: Clock | None = None) -> None:
        self.config = config
        self._clock = clock
        self._providers: dict[str, Provider] = {}
        self._lock = threading.Lock()

    # -- specs -----------------------------------------------------------

    def spec(self, name: str) -> ModelSpec:
        try:
            return self.config.models[name]
        except KeyError:
            raise ConfigError(f"unknown model {name!r}", known=sorted(self.config.models)) from None

    def names(self) -> list[str]:
        return list(self.config.models)

    @property
    def ladder(self) -> list[str]:
        return list(self.config.ladder)

    def ladder_specs(self) -> list[ModelSpec]:
        return [self.spec(name) for name in self.config.ladder]

    def rung(self, name: str) -> int:
        """Position on the escalation ladder; -1 for models not on it."""
        try:
            return self.config.ladder.index(name)
        except ValueError:
            return -1

    def next_rung(self, name: str) -> str | None:
        idx = self.rung(name)
        if idx < 0 or idx + 1 >= len(self.config.ladder):
            return None
        return self.config.ladder[idx + 1]

    # -- providers -------------------------------------------------------

    def provider_for(self, spec: ModelSpec) -> Provider:
        with self._lock:
            provider = self._providers.get(spec.provider)
            if provider is None:
                cfg = self.config.providers.get(spec.provider)
                if cfg is None:
                    raise ConfigError(f"model {spec.name!r} references undefined provider {spec.provider!r}")
                provider = build_provider(spec.provider, cfg, self._clock)
                self._providers[spec.provider] = provider
            return provider

    def install(self, name: str, provider: Provider) -> None:
        """Inject a provider instance. Used by tests and ``--dry-run``."""
        with self._lock:
            self._providers[name] = provider

    def available(self, name: str) -> bool:
        """Can this model be used right now (credentials present, etc.)?"""
        try:
            return self.provider_for(self.spec(name)).available()
        except ConfigError:
            return False

    def usable_ladder(self) -> list[str]:
        """The ladder filtered to models this host can actually reach.

        A box with no ``ANTHROPIC_API_KEY`` still runs; it simply has a shorter
        ladder and the router adapts. Degrading rather than refusing to start is
        the difference between a platform that survives five years of changing
        credentials and one that does not.
        """
        usable = [name for name in self.config.ladder if self.available(name)]
        if not usable:
            raise ConfigError(
                "no models are usable",
                ladder=self.config.ladder,
                hint="check the local model endpoint and any API-key environment variables",
            )
        return usable

    def describe(self) -> list[dict[str, object]]:
        rows = []
        for name in self.config.models:
            spec = self.spec(name)
            provider = self.config.providers.get(spec.provider)
            rows.append(
                {
                    "name": name,
                    "provider": spec.provider,
                    "provider_kind": provider.kind if provider else "?",
                    "model": spec.model,
                    "tier": spec.tier,
                    "hosted": spec.hosted,
                    "context": spec.context_window,
                    "max_output": spec.max_output_tokens,
                    "in_cost": spec.input_cost_per_mtok,
                    "out_cost": spec.output_cost_per_mtok,
                    "quota_per_hour": spec.quota_per_hour,
                    "thinking": spec.extra.get("thinking"),
                    "on_ladder": self.rung(name),
                    "available": self.available(name),
                }
            )
        return rows


def echo_registry(models: ModelsConfig | None = None) -> Registry:
    """A registry whose every provider is the deterministic echo stub."""
    from ..config import default_models

    config = models or default_models()
    for provider in config.providers.values():
        provider.kind = "echo"
        provider.api_key_env = ""
    registry = Registry(config)
    return registry


def make_provider_config(kind: str = "echo") -> ProviderConfig:  # pragma: no cover - convenience
    return ProviderConfig(kind=kind, base_url="", api_key_env="")
