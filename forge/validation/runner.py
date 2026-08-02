"""Running gates: ordering, parallelism, caching and reporting.

Three behaviours matter here, all of them about spending time well:

**Cheap gates first.** Gates carry an ``order``; a JSON parse check runs before
a browser boots. When ``fail_fast`` is on, the expensive gates are never reached
on a broken tree.

**Parallel where safe.** Gates that only read the tree run concurrently. Gates
that bind a port or drive a browser are serialised, because two dev servers
racing for the same port produce failures that look like application bugs and
cost hours to diagnose.

**Cached by content.** A cacheable gate whose declared inputs are unchanged
returns its previous verdict. Over a long run this is the difference between
re-running a two-minute test suite forty times a day and running it when
something it covers actually changed.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..errors import GateError
from ..kernel.events import Event, EventType
from ..kernel.ledger import Ledger
from ..obs.log import get_logger
from ..obs.metrics import Metrics
from ..util.clock import Clock, default_clock
from .gate import Gate, GateContext, gate_registry
from .types import ValidationReport, Verdict

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config

log = get_logger("validation.runner")

#: How much gate evidence to keep in the ledger event. Enough to name the cause
#: -- a compiler's first several errors -- without turning a long test log into a
#: database row. The full excerpt still reaches the repair prompt.
_LEDGER_EVIDENCE_CHARS = 2000

#: Gates that must not run concurrently with each other: they bind ports,
#: launch browsers, or otherwise contend for a global resource.
EXCLUSIVE = frozenset({"browser", "smoke", "load_perf", "benchmark", "integration"})

#: Durable home for the scripted user flow, so it outlives the node that
#: authored it and keeps verifying behaviour for the rest of the run.
SMOKE_FLOW_KEY = "validation.smoke_flow"


class GateRunner:
    def __init__(
        self,
        config: Config,
        ledger: Ledger,
        *,
        metrics: Metrics | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.metrics = metrics or Metrics(ledger, clock)
        self._clock = clock or default_clock()
        self._exclusive_lock = threading.Lock()

    # -- selection -------------------------------------------------------

    def resolve(self, names: list[str]) -> list[Gate]:
        gates: list[Gate] = []
        for name in names:
            try:
                gates.append(gate_registry.create(name))
            except Exception as exc:
                log.warn("unknown gate requested, skipping", gate=name, error=str(exc))
        gates.sort(key=lambda g: (g.order, g.name))
        return gates

    # -- execution -------------------------------------------------------

    def run(
        self,
        gates: list[str] | list[Gate],
        ctx: GateContext,
        *,
        fail_fast: bool | None = None,
        use_cache: bool | None = None,
    ) -> ValidationReport:
        # The union is `list[str] | list[Gate]`, so indexing one element decides
        # which. Casting keeps `resolved` a `list[Gate]`; without it every later
        # `.order`/`.name` access degrades to `object`.
        if gates and isinstance(gates[0], str):
            resolved = self.resolve(cast("list[str]", gates))
        else:
            resolved = cast("list[Gate]", list(gates))
        fail_fast = self.config.validation.fail_fast if fail_fast is None else fail_fast
        use_cache = self.config.validation.cache_results if use_cache is None else use_cache

        start = self._clock.monotonic()
        verdicts: list[Verdict] = []

        # Fail-fast has to operate between *phases*, not after submitting every
        # gate at once.  Exclusive gates used to run before everything else,
        # regardless of order, so Chromium and load-performance checks ran
        # before schema and TypeScript could report a syntax error.
        #
        # Shared gates in a phase remain parallel. Exclusive gates in that same
        # phase run serially under their global lock. The boundaries represent
        # increasing confidence/cost: static checks, build/tests, runtime
        # checks, then advisory quality measurements.
        phases = ((0, 39), (40, 99), (100, 129), (130, 10_000))
        for low, high in phases:
            phase = [gate for gate in resolved if low <= gate.order <= high]
            if not phase:
                continue
            shared = [gate for gate in phase if gate.name not in EXCLUSIVE]
            exclusive = [gate for gate in phase if gate.name in EXCLUSIVE]
            phase_pairs: list[tuple[Gate, Verdict]] = []
            if shared:
                workers = max(1, min(self.config.validation.parallel, len(shared)))
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="gate"
                ) as pool:
                    futures = {
                        gate: pool.submit(self._run_one, gate, ctx, use_cache=use_cache)
                        for gate in shared
                    }
                    phase_pairs.extend(
                        (gate, future.result()) for gate, future in futures.items()
                    )
            phase_pairs.extend(
                (gate, self._run_one(gate, ctx, use_cache=use_cache))
                for gate in exclusive
            )
            verdicts.extend(verdict for _, verdict in phase_pairs)
            if fail_fast and any(
                gate.blocking and not verdict.ok for gate, verdict in phase_pairs
            ):
                verdicts.sort(
                    key=lambda v: next(
                        (g.order for g in resolved if g.name == v.gate), 999
                    )
                )
                return self._finish(verdicts, start, ctx, aborted=True)

        verdicts.sort(key=lambda v: next((g.order for g in resolved if g.name == v.gate), 999))
        return self._finish(verdicts, start, ctx)

    def _finish(
        self, verdicts: list[Verdict], start: float, ctx: GateContext, *, aborted: bool = False
    ) -> ValidationReport:
        report = ValidationReport(verdicts=verdicts, duration=self._clock.monotonic() - start)
        blocking_failures = [
            v for v in verdicts if not v.ok and self._is_blocking(v.gate)
        ]
        report.verdicts = verdicts
        log.info(
            "validation complete",
            node=ctx.node_id,
            summary=report.summary_line(),
            blocking=len(blocking_failures),
            duration=round(report.duration, 2),
            aborted=aborted,
        )
        return report

    @staticmethod
    def _is_blocking(name: str) -> bool:
        try:
            return gate_registry.create(name).blocking
        except Exception:  # pragma: no cover
            return True

    def _run_one(self, gate: Gate, ctx: GateContext, *, use_cache: bool) -> Verdict:
        gate_ctx = GateContext(
            sandbox=ctx.sandbox,
            root=ctx.root,
            artifacts_dir=ctx.artifacts_dir,
            toolchain=ctx.toolchain,
            settings=dict(ctx.settings.get(gate.name, {})) if isinstance(ctx.settings.get(gate.name), dict) else {},
            changed_files=ctx.changed_files,
            node_id=ctx.node_id,
            timeout=ctx.timeout,
            memory=ctx.memory,
        )

        if not gate.applicable(gate_ctx):
            verdict = Verdict.skip(gate.name, "not applicable to this project")
            self._emit(verdict, gate_ctx, cache_key="")
            return verdict

        cache_key = ""
        if use_cache and gate.cacheable:
            try:
                cache_key = gate.cache_key(gate_ctx)
            except OSError as exc:  # pragma: no cover - tree read failure
                log.warn("could not compute gate cache key", gate=gate.name, error=str(exc))
            if cache_key:
                cached = self._cached(cache_key)
                if cached is not None:
                    self.metrics.incr("gate.cache_hit", gate=gate.name)
                    log.debug("gate result served from cache", gate=gate.name)
                    return cached

        self.ledger.append(
            Event(type=EventType.GATE_STARTED, node_id=gate_ctx.node_id, payload={"gate": gate.name})
        )

        lock = self._exclusive_lock if gate.name in EXCLUSIVE else _NullLock()
        started = self._clock.monotonic()
        with lock:
            try:
                verdict = gate.run(gate_ctx)
            except GateError as exc:
                verdict = Verdict.error(gate.name, str(exc))
            except Exception as exc:  # a broken gate must not kill the run
                log.exception("gate raised", exc, gate=gate.name)
                verdict = Verdict.error(gate.name, f"{type(exc).__name__}: {exc}")
        verdict.duration = verdict.duration or (self._clock.monotonic() - started)

        self.metrics.observe("gate.duration", verdict.duration, gate=gate.name)
        self.metrics.incr(
            "gate.result",
            gate=gate.name,
            outcome="pass" if verdict.ok else ("error" if verdict.errored else "fail"),
        )
        self._emit(verdict, gate_ctx, cache_key=cache_key if gate.cacheable else "")
        return verdict

    def _emit(self, verdict: Verdict, ctx: GateContext, *, cache_key: str) -> None:
        if verdict.skipped:
            event_type = EventType.GATE_SKIPPED
        elif verdict.errored:
            event_type = EventType.GATE_ERRORED
        elif verdict.passed:
            event_type = EventType.GATE_PASSED
        else:
            event_type = EventType.GATE_FAILED
        payload = {
            "gate": verdict.gate,
            "cache_key": cache_key,
            "summary": verdict.summary,
            "score": verdict.score,
            "duration": verdict.duration,
            "issue_count": len(verdict.issues),
            "artifacts": verdict.artifacts,
            "detail": verdict.detail,
        }
        # Why it failed, not just that it did. The evidence is already assembled
        # and already bounded -- `render` feeds it to the repair prompt -- so the
        # model could see the reason while an operator reading the ledger got
        # `{"exit_code": 2, "command": "npm run typecheck"}` and nothing else.
        # Finding out that a source file was zero bytes meant leaving the tooling
        # and running tsc by hand. Recorded only for failures: a pass has nothing
        # to explain, and every gate on every node would bloat the ledger.
        if event_type in (EventType.GATE_FAILED, EventType.GATE_ERRORED):
            evidence = (verdict.evidence or "").strip()
            if evidence:
                payload["evidence"] = evidence[:_LEDGER_EVIDENCE_CHARS]
        self.ledger.append(Event(type=event_type, node_id=ctx.node_id, payload=payload))

    def _cached(self, cache_key: str) -> Verdict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM gate_results WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        # Only *passes* are reused. A cached failure would be reused after the
        # fix if the fix touched a file outside the gate's declared inputs --
        # rare, but the failure mode (a permanently red gate) is bad enough that
        # re-running failures is worth the time.
        if not row["passed"]:
            return None
        return Verdict(
            gate=row["gate"],
            passed=True,
            summary=row["summary"] or "",
            score=row["score"],
            duration=row["duration"],
            cached=True,
        )

    # -- context construction -------------------------------------------

    def build_context(
        self,
        *,
        root: Path,
        sandbox: Any,
        toolchain: dict[str, Any],
        node_id: str | None = None,
        changed_files: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        memory: Any = None,
    ) -> GateContext:
        merged: dict[str, Any] = {
            "visual": {"tolerance": self.config.validation.visual_tolerance},
            "browser": {
                "headless": self.config.validation.browser_headless,
            },
            "coverage": {"floor": self.config.validation.coverage_floor},
        }
        # Project configuration sits between the built-in defaults and whatever
        # the node supplies, so a project can set routes and budgets once while a
        # node stays free to override them.
        for key, value in self.config.validation.gate_settings.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        for key, value in (settings or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        # The smoke gate verifies behaviour rather than compilation, and it is
        # the only gate that does. It is applicable only when someone hands it
        # steps, and the agent that authors them wrote them into a single
        # node's spec -- so on a real project it ran twice and skipped 144
        # times with "not applicable to this project" while every other
        # validation happily passed a game nobody could play.
        #
        # Once a flow has been authored it describes the product, not that one
        # node, so fall back to the durable copy whenever a caller has not
        # supplied its own.
        if not (merged.get("smoke") or {}).get("steps"):
            stored = self.ledger.kv_get(SMOKE_FLOW_KEY) or {}
            steps = stored.get("steps") if isinstance(stored, dict) else None
            if steps:
                merged["smoke"] = {**merged.get("smoke", {}), "steps": steps}
        return GateContext(
            sandbox=sandbox,
            root=root,
            artifacts_dir=self.config.artifacts_dir,
            toolchain=toolchain,
            settings=merged,
            changed_files=changed_files or [],
            node_id=node_id,
            timeout=self.config.sandbox.command_timeout,
            # Was `self.config.validation.browser_timeout if False else ...`,
            # which made `browser_timeout` dead config: setting it changed
            # nothing, and `page.goto` inherited the 900s command ceiling, so a
            # page that never loads hung the gate for fifteen minutes.
            browser_timeout=self.config.validation.browser_timeout,
            memory=memory,
        )


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
