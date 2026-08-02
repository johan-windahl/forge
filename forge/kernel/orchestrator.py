"""The orchestrator: the long-running loop that builds the software.

This is where every subsystem meets. Its responsibilities, in order of
importance for unattended operation:

1. **Own the durability boundary.** An agent returns a description of what it
   did; the orchestrator is the only thing that makes it real -- commit, write
   memory, extend the graph, mark the node -- and it does so in a fixed order
   chosen so that a crash at any point leaves a recoverable state.
2. **Keep working.** Nothing an agent, a model or a gate does may stop the loop.
   Every failure becomes a scheduling decision, and the loop moves to the next
   node.
3. **Be resumable.** Starting the orchestrator on an existing project resumes
   exactly where it left off, including nodes that were mid-flight when the
   process died.

Concurrency is threads, not processes or async. The work is dominated by waiting
on subprocesses and HTTP, threads make the SQLite and git interactions
straightforward, and the number of workers is small. Nothing in the design
prevents a future distributed executor -- leases and event sourcing were chosen
partly for that -- but adding it now would buy complexity, not throughput.
"""

from __future__ import annotations

import hashlib
import re
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.base import AgentContext, AgentResult, ProposedNode
from ..agents.registry import build_agent
from ..config import Config
from ..errors import (
    Abort,
    BudgetExhausted,
    ForgeError,
    GitError,
    HumanInputRequired,
    LeaseLost,
    MergeConflict,
    NotSupported,
    ValidationFailed,
)
from ..memory.lessons import LessonLibrary
from ..memory.records import MemoryKind
from ..memory.store import MemoryStore, requirement
from ..models.client import ModelClient
from ..models.health import probe_provider
from ..obs.log import get_logger
from ..obs.metrics import Metrics
from ..util.clock import Clock, default_clock, human_duration
from ..util.ids import new_id
from ..validation.runner import SMOKE_FLOW_KEY, GateRunner
from ..workspace.deps import STAMP_PATH
from ..workspace.deps import ensure as ensure_dependencies
from ..workspace.git import Repo
from ..workspace.sandbox import Sandbox, build_sandbox, detect_toolchain
from .checkpoint import CheckpointManager
from .events import Event, EventType
from .graph import Node, NodeKind, NodeStatus, TaskGraph
from .ledger import Ledger
from .scheduler import Disposition, Scheduler

log = get_logger("kernel.orchestrator")

#: Consecutive idle polls before the loop concludes it cannot progress.
IDLE_POLLS_BEFORE_STOP = 5
# Native coding agents currently have five bounded model/fix rounds. Keeping
# the drain calculation local avoids importing the coding registry back into
# the kernel and creating an agents -> kernel -> agents cycle.
MAX_AGENT_MODEL_ROUNDS = 5


@dataclass(slots=True)
class RunStats:
    started_at: float = 0.0
    nodes_completed: int = 0
    nodes_failed: int = 0
    nodes_blocked: int = 0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "nodes_completed": self.nodes_completed,
            "nodes_failed": self.nodes_failed,
            "nodes_blocked": self.nodes_blocked,
            "attempts": self.attempts,
        }


@dataclass(slots=True)
class Project:
    """Identity and top-level state of one build."""

    id: str
    name: str
    goal: str
    created_at: float = 0.0
    root_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "created_at": self.created_at,
            "root_node": self.root_node,
            "metadata": self.metadata,
        }


class Orchestrator:
    """Runs a project to completion, or until told to stop."""

    def __init__(self, config: Config, *, clock: Clock | None = None, worker_prefix: str = "forge") -> None:
        self.config = config
        self._clock = clock or default_clock()
        config.ensure_dirs()

        self.ledger = Ledger(config.ledger_path, clock=self._clock)
        self.metrics = Metrics(self.ledger, self._clock)
        self.project = self._load_or_none()
        project_id = self.project.id if self.project else ""
        self.ledger.project_id = project_id

        self.graph = TaskGraph(self.ledger, project_id, self._clock)
        self.memory = MemoryStore(self.ledger, project_id, self._clock)
        self.lessons = LessonLibrary(
            Path(config.memory.lessons_global_path).expanduser(),
            clock=self._clock,
            project=self.project.name if self.project else "",
        )
        self.models = ModelClient(config, self.ledger, metrics=self.metrics, clock=self._clock)
        self.gates = GateRunner(config, self.ledger, metrics=self.metrics, clock=self._clock)
        #: Set when every provider has been swapped for the echo stub. Recorded
        #: on the run so that progress from a rehearsal is never mistaken for
        #: progress on the goal.
        self.dry_run = False

        self.repo = Repo(config.workspace_dir)
        self.sandbox: Sandbox = build_sandbox(config.sandbox, config.workspace_dir)
        self.checkpoints = CheckpointManager(self.ledger, self.repo, self._clock, graph=self.graph)
        self.scheduler = Scheduler(
            self.graph,
            config.scheduler,
            clock=self._clock,
            ladder=config.models.ladder,
            per_node_cost=config.budget.per_node_cost,
        )

        self.worker_prefix = worker_prefix
        self._stop = threading.Event()
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()
        # Node work is isolated in branches, but integration into main is a
        # short serial transaction: merge, validate, checkpoint.
        self._integration_lock = threading.RLock()
        self._toolchain: dict[str, Any] = {}
        #: Per-model (calls, in, out, cost) at the last usage report, so each
        #: report can show the window rather than only the running total.
        self._usage_mark: dict[str, tuple[int, int, int, float]] = {}
        #: Latched so an unreachable endpoint is reported once per outage rather
        #: than once per heartbeat.
        self._endpoint_unreachable = False
        self._digest: str = ""
        self._digest_version = -1
        self.stats = RunStats()

        # Registering built-in gates has to happen before any node runs; the
        # import has the side effect of populating the registry.
        from ..validation import gates as _gates  # noqa: F401

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def _load_or_none(self) -> Project | None:
        data = self.ledger.kv_get("project")
        return Project(**data) if data else None

    def create_project(self, goal: str, *, name: str = "") -> Project:
        """Initialise a new project from a single goal statement.

        Everything the human provides ends here. From this point the platform
        decides what to build, in what order, and how to verify it.
        """
        if self.project is not None:
            raise NotSupported(
                "a project already exists in this directory",
                project=self.project.name,
                hint="use a different --dir, or `forge reset`",
            )
        project = Project(
            id=new_id("proj"),
            name=name or _slug(goal),
            goal=goal.strip(),
            created_at=self._clock.now(),
        )
        self.ledger.project_id = project.id
        self.graph.project_id = project.id
        self.memory.project_id = project.id
        self.ledger.kv_set("project", project.to_dict())
        self.ledger.emit(EventType.PROJECT_CREATED, **project.to_dict())

        self.repo.init()
        self.memory.write(
            requirement(
                "Project goal",
                goal.strip(),
                source="human",
                tags=["goal"],
            )
        )

        root = self.graph.add_node(
            NodeKind.GOAL,
            f"Deliver: {goal.strip()[:90]}",
            spec={
                "objective": goal.strip(),
                "acceptance": ["The stated goal is delivered and verified"],
                # A barrier: runnable only once nothing else can progress, which
                # is the only point at which "is this finished?" is answerable.
                "barrier": True,
                "check_round": 0,
            },
            priority=1000,
            actor="human",
        )
        plan = self.graph.add_node(
            NodeKind.PLAN,
            "Plan the project",
            spec={
                "objective": "Decompose the goal into milestones and an executable task graph",
                "acceptance": ["A task graph exists for the first milestone"],
            },
            parent_id=root.id,
            priority=1,
            actor="human",
        )
        project.root_node = root.id
        project.metadata["plan_node"] = plan.id
        self.ledger.kv_set("project", project.to_dict())
        self.project = project

        self.checkpoints.create("project created", kind="milestone", metadata={"goal": goal})
        log.info("project created", project=project.name, id=project.id)
        return project

    # ------------------------------------------------------------------
    # The run loop
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        max_nodes: int | None = None,
        until_quiescent: bool = True,
        install_signal_handlers: bool = True,
    ) -> dict[str, Any]:
        """Run until the graph is quiescent, a limit is hit, or stopped.

        Returns a summary. Safe to call repeatedly: a second call resumes.
        """
        if self.project is None:
            raise NotSupported("no project in this directory; run `forge init` first")

        self.stats = RunStats(started_at=self._clock.now())
        self._stop.clear()
        if install_signal_handlers:
            self._install_signals()

        self._recover()
        self._refresh_toolchain()
        self._reopen_if_unfinished()

        run_id = new_id("run")
        self.ledger.emit(
            EventType.RUN_STARTED,
            run_id=run_id,
            workers=self.config.scheduler.workers,
            ladder=self.config.models.ladder,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            log.warn(
                "dry run: nodes will 'succeed' on stub output. Progress recorded "
                "by this run does not mean the goal was built."
            )
        log.info(
            "run started",
            project=self.project.name,
            workers=self.config.scheduler.workers,
            budget=self.config.budget.total_cost,
        )

        heartbeat = threading.Thread(target=self._heartbeat, args=(run_id,), daemon=True, name="forge-heartbeat")
        heartbeat.start()

        completed = 0
        idle_polls = 0
        stuck: list[str] = []
        workers = max(1, self.config.scheduler.workers)
        # Not a `with` block: leaving one calls shutdown(wait=True), which blocks
        # forever on a worker that is stuck in a call it cannot be interrupted
        # out of. Shutdown is done explicitly below so it can be bounded.
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=self.worker_prefix)
        try:
            futures: dict[Any, str] = {}
            while not self._stop.is_set():
                if max_nodes is not None and completed >= max_nodes:
                    break

                # Harvest finished work first so slots free up promptly.
                for future in [f for f in futures if f.done()]:
                    futures.pop(future)
                    completed += 1
                    try:
                        future.result()
                    except Exception as exc:  # pragma: no cover - defence in depth
                        log.exception("worker raised", exc)

                if len(futures) < workers:
                    node = self._pick_node()
                    if node is not None:
                        idle_polls = 0
                        worker_id = f"{self.worker_prefix}-{new_id('w')[-6:]}"
                        futures[pool.submit(self._execute, node, worker_id)] = node.id
                        continue

                if not futures:
                    self.scheduler.sweep()
                    if until_quiescent and self.graph.is_quiescent():
                        break
                    if self._budget_stopped():
                        break
                    # Require the idle condition to persist. Checking it once
                    # races with backoff timers: a node whose retry is due in
                    # a millisecond is neither runnable *now* nor waiting on a
                    # future wake-up, and a single-sample check would call
                    # that a permanent stall.
                    idle_polls = idle_polls + 1 if self._idle_now() else 0
                    if idle_polls >= IDLE_POLLS_BEFORE_STOP:
                        log.warn(
                            "nothing is runnable and no timer will fire; stopping",
                            counts=self.graph.counts(),
                        )
                        break
                else:
                    idle_polls = 0
                self._clock.sleep(self.config.scheduler.poll_interval)

            stuck = self._drain(futures, pool)
        except Abort:
            log.info("run aborted by operator")
        finally:
            self._stop.set()
            heartbeat.join(timeout=2)
            self.checkpoints.create("run stopped", kind="manual")
            self.sandbox.teardown()

        summary = self.status()
        self.ledger.emit(
            EventType.RUN_STOPPED,
            run_id=run_id,
            duration=self._clock.now() - self.stats.started_at,
            stuck_nodes=stuck,
            **self.stats.to_dict(),
        )
        log.info(
            "run finished",
            completed=self.stats.nodes_completed,
            blocked=self.stats.nodes_blocked,
            duration=human_duration(self._clock.now() - self.stats.started_at),
            cost=summary["budget"]["total"],
        )
        if stuck:
            # Every durable write is done: the checkpoint exists, run.stopped is
            # recorded. What remains is a non-daemon thread that will never
            # return, and waiting for it would leave a process that looks alive
            # and does nothing -- the failure mode this whole path exists to end.
            self._force_exit(130)
        return summary

    def stop(self) -> None:
        """Ask the run to wind down after in-flight nodes finish."""
        self._stop.set()

    def _drain(self, futures: dict[Any, str], pool: ThreadPoolExecutor) -> list[str]:
        """Wait for in-flight nodes, bounded. Returns the ids still running.

        Unbounded here means unstoppable: a worker blocked on a call it cannot be
        interrupted out of would keep the process alive indefinitely, which is
        what turned "Ctrl-C" into "kill -9" in practice. The grace period is
        generous enough for a node that is genuinely finishing, and after it the
        node's lease simply expires and the next run re-runs the attempt --
        exactly the path recovery already handles for a machine that lost power.
        """
        grace = self._drain_grace(futures)
        deadline = self._clock.monotonic() + grace
        for future in list(futures):
            remaining = max(0.0, deadline - self._clock.monotonic())
            try:
                future.result(timeout=remaining)
            except TimeoutError:
                continue
            except Exception as exc:  # pragma: no cover - defence in depth
                log.exception("worker raised during shutdown", exc)
        stuck = [node_id for future, node_id in futures.items() if not future.done()]
        # cancel_futures clears the queue; threads already running cannot be
        # cancelled, which is precisely why `stuck` has to be reported upwards.
        pool.shutdown(wait=not stuck, cancel_futures=True)
        if stuck:
            log.warn(
                "in-flight node(s) did not finish within the shutdown grace; "
                "their leases will expire and the next run re-runs them",
                grace=grace,
                nodes=[n[-8:] for n in stuck],
            )
        return stuck

    def _drain_grace(self, futures: dict[Any, str]) -> float:
        """Return the real completion window for an operator-requested drain.

        A two-minute constant is not graceful when one advertised local call
        may take an hour and an OpenCode node may run several tool rounds. Zero
        means automatic: cover the remaining bounded agent loop plus validation.
        Operators who need a short bound can set an explicit positive grace,
        and a second signal still exits immediately.
        """
        configured = float(self.config.scheduler.shutdown_grace)
        if configured > 0:
            return configured

        longest = 0.0
        for node_id in futures.values():
            node = self.graph.try_get(node_id)
            if node is None:
                continue
            spec = self.config.models.models.get(node.tier)
            if spec is None:
                continue
            if spec.hosted == "local" and self.config.coding.backend != "native":
                attempt = (
                    self.config.coding.opencode_timeout
                    * self.config.coding.opencode_rounds
                )
            else:
                attempt = spec.timeout * MAX_AGENT_MODEL_ROUNDS
            longest = max(longest, attempt)
        validation = max(
            300.0,
            self.config.sandbox.command_timeout,
            self.config.validation.browser_timeout,
        )
        return longest + validation

    def _install_signals(self) -> None:
        """First signal winds down; a second one exits now.

        An operator who presses Ctrl-C twice means "now", and is right to: the
        first request can legitimately take minutes while a node finishes, and
        with the handler installed there is no longer a KeyboardInterrupt to fall
        back on. Without this escalation the only remaining option is `kill -9`
        from another terminal, which is a worse outcome than an unclean stop --
        the ledger is append-only and recovery is designed for exactly this.
        """
        signals = {"count": 0}

        def handler(signum: int, _frame: Any) -> None:
            signals["count"] += 1
            if signals["count"] == 1:
                log.warn("signal received, winding down; press again to exit now", signal=signum)
                self._stop.set()
                return
            log.warn("second signal: exiting immediately", signal=signum)
            self._force_exit(130)

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:  # pragma: no cover - not on the main thread
            log.debug("signal handlers not installed (not main thread)")

    def _force_exit(self, code: int) -> None:  # pragma: no cover - replaced in tests
        """Leave without waiting for threads that cannot be interrupted.

        ``os._exit`` skips interpreter shutdown, which is the point: the normal
        path joins non-daemon worker threads and would hang. Everything that
        matters is already durable in the ledger.
        """
        import os

        from ..util.proc import terminate_active_processes

        terminate_active_processes()
        self.ledger.close()
        os._exit(code)

    def _heartbeat(self, run_id: str) -> None:
        """Prove liveness and record progress while the run is long.

        A heartbeat in the ledger is what lets an operator -- or a supervising
        process -- tell "still working" from "hung" without attaching a debugger
        to a three-day-old process.
        """
        interval = self.config.scheduler.heartbeat_interval
        next_usage = self._clock.monotonic() + self.config.scheduler.usage_report_interval
        while not self._stop.wait(interval):
            try:
                counts = self.graph.counts()
                self.ledger.emit(
                    EventType.RUN_HEARTBEAT,
                    run_id=run_id,
                    counts=counts,
                    progress=round(self.graph.progress(), 4),
                    in_flight=sorted(self._in_flight),
                    cost=round(self.models.budget.snapshot().total, 4),
                )
                self.scheduler.sweep()
                every = self.config.scheduler.usage_report_interval
                if every > 0 and self._clock.monotonic() >= next_usage:
                    next_usage = self._clock.monotonic() + every
                    self._report_usage(run_id)
            except Exception as exc:  # pragma: no cover - heartbeat must not die
                log.warn("heartbeat failed", error=str(exc))

    def _report_usage(self, run_id: str) -> None:
        """Log token usage per model, and what changed since the last report.

        The delta is the point. Cumulative totals answer "what has this project
        cost", which `forge status` already does; they cannot tell you whether
        the last five minutes went to the local rung or to a frontier one, which
        is the question an operator watching a long run actually has.
        """
        by_model = {
            str(row["model"]): row for row in self.models.budget.report().get("by_model", [])
        }
        deltas: list[dict[str, Any]] = []
        for name, row in by_model.items():
            before = self._usage_mark.get(name, (0, 0, 0, 0.0))
            entry = {
                "model": name,
                "hosted": row.get("hosted", ""),
                "calls": int(row.get("calls") or 0) - before[0],
                "input_tokens": int(row.get("input_tokens") or 0) - before[1],
                "output_tokens": int(row.get("output_tokens") or 0) - before[2],
                "cost": round(float(row.get("cost") or 0.0) - before[3], 4),
                "total_input_tokens": int(row.get("input_tokens") or 0),
                "total_output_tokens": int(row.get("output_tokens") or 0),
                "total_cost": round(float(row.get("cost") or 0.0), 4),
            }
            self._usage_mark[name] = (
                int(row.get("calls") or 0),
                int(row.get("input_tokens") or 0),
                int(row.get("output_tokens") or 0),
                float(row.get("cost") or 0.0),
            )
            deltas.append(entry)

        deltas.sort(key=lambda d: -d["output_tokens"])
        snapshot = self.models.budget.snapshot()
        total_tokens = snapshot.local_tokens + snapshot.cloud_tokens
        self.ledger.emit(
            EventType.USAGE_REPORT,
            run_id=run_id,
            window_seconds=self.config.scheduler.usage_report_interval,
            models=deltas,
            cloud_fraction=round(snapshot.cloud_tokens / total_tokens, 4) if total_tokens else 0.0,
            total_cost=round(snapshot.total, 4),
        )
        active = [d for d in deltas if d["calls"]]
        if not active:
            log.info("usage: no model calls in the last window")
            self._check_silent_window(run_id)
            return
        # Calls landed, so whatever was serving them is reachable by definition.
        self._clear_endpoint_warning(run_id, "model calls resumed")
        for entry in active:
            log.info(
                "usage",
                model=entry["model"],
                hosted=entry["hosted"],
                calls=entry["calls"],
                tokens_in=entry["input_tokens"],
                tokens_out=entry["output_tokens"],
                cost=entry["cost"],
                total_out=entry["total_output_tokens"],
            )

    def _check_silent_window(self, run_id: str) -> None:
        """Explain a quiet usage window when a node is supposed to be working.

        A window with no model calls is normal when nothing is running. With a
        node running it is not: either the model is mid-generation, or the run is
        blocked on an endpoint that will never answer. The two look identical
        from here, so ask the endpoint directly.

        This exists because an unreachable local server is silent by nature. The
        inner executor blocks on its socket, the node keeps its lease, and the
        heartbeat keeps reporting "no model calls" without ever saying why. One
        run sat that way for close to two hours after its base URL changed under
        it. The probe is free; the silence was not.
        """
        if not self.graph.counts().get(NodeStatus.RUNNING, 0):
            self._endpoint_unreachable = False
            return

        provider = self.config.models.providers.get("local")
        if provider is None or provider.kind not in {"openai_compat", "openai", "anthropic"}:
            return

        reachable, detail = probe_provider(provider)
        if reachable:
            self._clear_endpoint_warning(run_id, detail)
            return

        # Log the transition loudly, then stay quiet: the heartbeat runs every
        # few minutes and a repeating error teaches an operator to ignore it.
        # The *event* is emitted every window regardless -- the ledger is state,
        # not a log, and `forge status` reads the newest one to decide whether
        # the outage is still current.
        if not self._endpoint_unreachable:
            log.error(
                "local model endpoint unreachable while a node is running",
                detail=detail,
                hint="check the server, or set [models.providers.local].base_url "
                "(or FORGE_LOCAL_BASE_URL)",
            )
        self.ledger.emit(
            EventType.RUN_WARNING,
            run_id=run_id,
            kind="model_endpoint_unreachable",
            provider="local",
            detail=detail,
            resolved=False,
        )
        self._endpoint_unreachable = True

    def _clear_endpoint_warning(self, run_id: str, detail: str = "") -> None:
        """Record that the endpoint is answering again.

        Emitted rather than inferred. Letting recovery be implied by "some other
        event was written afterwards" makes whether the stall still shows depend
        on write ordering elsewhere in the heartbeat.
        """
        if not self._endpoint_unreachable:
            return
        log.info("local model endpoint is answering again", detail=detail)
        self.ledger.emit(
            EventType.RUN_WARNING,
            run_id=run_id,
            kind="model_endpoint_unreachable",
            provider="local",
            detail=detail,
            resolved=True,
        )
        self._endpoint_unreachable = False

    def _idle_now(self) -> bool:
        """One sample of "nothing to do and nothing coming".

        The graph reports non-quiescent (nodes remain), yet nothing is runnable
        and no deferred timer will fire. Sustained across several polls this
        means the orchestrator would spin forever on a graph it cannot advance
        -- burning a core and, worse, never telling the operator it is stuck.
        """
        if self.graph.runnable(limit=1):
            return False
        if self.graph.next_wakeup() is not None:
            return False
        with self._in_flight_lock:
            return not self._in_flight

    def _budget_stopped(self) -> bool:
        if not self.config.budget.stop_on_exhaustion:
            return False
        snapshot = self.models.budget.snapshot()
        if (
            snapshot.cloud_tokens + snapshot.local_tokens > 0
            and snapshot.cloud_fraction > self.config.budget.max_cloud_fraction
        ):
            log.error(
                "hard cloud-generated-token ceiling crossed, stopping run",
                fraction=round(snapshot.cloud_fraction, 4),
            )
            self.ledger.emit(
                EventType.BUDGET_EXHAUSTED,
                cloud_fraction=snapshot.cloud_fraction,
                limit=self.config.budget.max_cloud_fraction,
            )
            return True
        if (
            self.config.budget.enforce_cost_limits
            and snapshot.total >= self.config.budget.total_cost
        ):
            log.error("budget exhausted, stopping run", spent=round(snapshot.total, 4))
            self.ledger.emit(EventType.BUDGET_EXHAUSTED, spent=snapshot.total, limit=self.config.budget.total_cost)
            return True
        return False

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover(self) -> None:
        """Restore a consistent state after a crash or a clean stop.

        Three things can be wrong on startup: leases held by a dead process,
        nodes stuck in ``pending`` whose dependencies actually completed, and a
        dirty working tree from an interrupted attempt. All three are repaired
        without human involvement, which is the whole point.
        """
        self.repo.ensure_private_excludes()

        # Order matters: release the leases first, then repair node status.
        # Doing it the other way round leaves a READY node pinned by a lease
        # nobody holds, which no later sweep would ever clear.
        reaped = self.graph.release_all_leases()
        promoted = self.graph.promote_ready()

        stale = [n for n in self.graph.all_nodes(status=NodeStatus.RUNNING)]
        for node in stale:
            self.graph.update(node.id, status=str(NodeStatus.READY), actor="recovery")

        dirty = False
        if self.repo.exists and self.repo.is_dirty():
            dirty = True
            self.checkpoints.restore_for_attempt("recovery")

        if reaped or promoted or stale or dirty:
            log.info(
                "recovered previous state",
                reclaimed_leases=len(reaped),
                promoted=len(promoted),
                reset_running=len(stale),
                workspace_reset=dirty,
            )

    def _refresh_toolchain(self) -> None:
        self._toolchain = detect_toolchain(self.sandbox)
        if self._toolchain.get("languages"):
            log.info(
                "toolchain detected",
                languages=self._toolchain["languages"],
                commands=list(self._toolchain.get("commands", {})),
            )
        self._ensure_dependencies()

    def _harvest_escalations(self) -> None:
        """Add this project's escalation pairs to the cross-project corpus.

        Run at milestone boundaries rather than per node: the pairs are read
        from the ledger, so nothing is lost by batching, and a milestone is the
        point at which a node's outcome has actually settled.

        Deliberately capture-only. Turning a pair into a rule needs a corpus
        large enough to tell a general failure mode from one bad afternoon, and
        needs to exclude pairs where the "local failure" was really a platform
        defect -- which is why the rejecting evidence travels with the pair.
        """
        from ..improve.escalation import EscalationCorpus, find_pairs

        try:
            titles = {n.id: n.title for n in self.graph.all_nodes()}
            pairs = find_pairs(
                self.ledger,
                ladder=self.config.models.ladder,
                project=self.project.name if self.project else "",
                titles=titles,
            )
            corpus = EscalationCorpus(self.config.memory.escalations_global_path)
            written = corpus.record(pairs)
            if written:
                log.info("escalation pairs harvested", new=written, total=len(pairs))
        except Exception as exc:  # never let bookkeeping stop a build
            log.warn("could not harvest escalation pairs", error=str(exc))

    def _ensure_dependencies(self) -> None:
        """Install dependencies when the manifest says they are needed.

        Called after any node that touches a manifest, because a scaffolded
        project's dependencies do not exist until something fetches them, and
        until they do every gate either skips or -- worse -- reports the missing
        toolchain as a defect in the generated code.
        """
        record = ensure_dependencies(
            self.sandbox,
            self._toolchain,
            timeout=self.config.sandbox.install_timeout,
            enabled=self.config.sandbox.install_dependencies,
        )
        if record is None:
            return
        self.ledger.emit(EventType.TOOLCHAIN_INSTALLED, **record)
        if record.get("ok"):
            # New binaries mean new runnable gates; re-detect so they are used.
            self._toolchain = detect_toolchain(self.sandbox)

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    def _pick_node(self) -> Node | None:
        with self._in_flight_lock:
            excluded = set(self._in_flight)
        node = self.scheduler.next_node(exclude=excluded)
        if node is None:
            return None
        with self._in_flight_lock:
            if node.id in self._in_flight:
                return None
            self._in_flight.add(node.id)
        return node

    def _release(self, node_id: str) -> None:
        with self._in_flight_lock:
            self._in_flight.discard(node_id)

    def _execute(self, node: Node, worker_id: str) -> None:
        """Run one node attempt end to end."""
        logger = log.bind(node=node.id, kind=node.kind, worker=worker_id)
        lease = None
        renewer: threading.Thread | None = None
        stop_renew = threading.Event()
        attempt_repo = self.repo
        attempt_sandbox = self.sandbox
        isolated = False

        try:
            try:
                lease = self.graph.claim(node.id, worker_id, self.config.scheduler.lease_seconds)
            except ForgeError as exc:
                logger.debug("could not claim node", error=str(exc))
                return

            renewer = threading.Thread(
                target=self._renew_lease, args=(lease, stop_renew), daemon=True, name=f"lease-{node.id[-6:]}"
            )
            renewer.start()

            self.stats.attempts += 1
            node = self.graph.start(node.id, tier=node.tier, worker_id=worker_id)
            logger.info("node started", title=node.title, attempt=node.attempts, tier=node.tier)

            agent = build_agent(node.kind)
            if agent.commits:
                attempt_repo, attempt_sandbox = self._node_workspace(node)
                isolated = True
            else:
                # Read-only agents see only integrated, validated project state.
                self.checkpoints.restore_for_attempt(node.id)

            with self.metrics.timer("node.duration", kind=node.kind):
                result = self._run_agent(
                    node,
                    agent=agent,
                    repo=attempt_repo,
                    sandbox=attempt_sandbox,
                )

            self._commit_result(
                node,
                result,
                lease,
                repo=attempt_repo,
                sandbox=attempt_sandbox,
            )

        except LeaseLost as exc:
            logger.warn("lost lease mid-flight; another worker will take over", error=str(exc))
        except BudgetExhausted as exc:
            logger.error("budget exhausted during node", error=str(exc))
            self.graph.block(node.id, "budget exhausted", question=str(exc))
            self.stats.nodes_blocked += 1
            self._stop.set()
        except HumanInputRequired as exc:
            logger.warn("node needs human input", error=str(exc))
            self.graph.block(node.id, exc.message, question=str(exc))
            self.stats.nodes_blocked += 1
        except Exception as exc:
            logger.exception("node attempt failed", exc)
            self._handle_failure(node, exc, repo=attempt_repo)
        finally:
            stop_renew.set()
            if renewer is not None:
                renewer.join(timeout=2)
            if lease is not None:
                self.graph.release(lease)
            self._release(node.id)
            if isolated and attempt_sandbox is not self.sandbox:
                attempt_sandbox.teardown()
            finished = self.graph.try_get(node.id)
            if isolated and finished is not None and finished.status == NodeStatus.SUCCEEDED:
                branch = attempt_repo.branch()
                self.repo.remove_worktree(attempt_repo.path)
                self.repo.delete_branch(branch)

    def _renew_lease(self, lease: Any, stop: threading.Event) -> None:
        interval = self.config.scheduler.lease_renew_interval
        while not stop.wait(interval):
            try:
                self.graph.renew(lease, self.config.scheduler.lease_seconds)
            except LeaseLost:
                log.warn("lease lost while renewing", node=lease.node_id)
                return
            except Exception as exc:  # pragma: no cover
                log.warn("lease renewal failed", node=lease.node_id, error=str(exc))

    def _reopen_if_unfinished(self) -> None:
        """Re-open a drained project that still has unresolved review findings.

        A graph with nothing left to do is quiescent, and `forge run` on a
        quiescent graph returns immediately. That is correct when the project
        is finished and a dead end when it is not: a project was declared
        complete at 100% while four critical review findings were open, and
        after that there was no supported way back in. `forge tell` writes a
        memory record but creates no node, so nothing ever read it. The only
        remaining option was editing the ledger by hand.

        So: if there is no actionable work but a reviewer's unresolved verdict
        still contradicts "delivered", queue one goal re-check. The goal agent
        already knows how to turn those findings into work, and it stops on its
        own when they are closed or when they stop converging.
        """
        from ..agents.goal import _blocking_findings

        actionable = [
            node
            for node in self.graph.all_nodes()
            if node.status in (NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING)
        ]
        if actionable:
            return
        blocking = _blocking_findings(self.memory.open_findings())
        if not blocking:
            return

        self.graph.add_node(
            NodeKind.GOAL,
            "Re-check the goal against unresolved review findings",
            spec={
                "objective": (
                    "Judge whether the project delivers the stated goal. "
                    f"{len(blocking)} review finding(s) were never resolved."
                ),
                "acceptance": ["The stated goal is delivered and verified"],
                "barrier": True,
                "reopened": True,
            },
            priority=990,
            actor="orchestrator",
        )
        log.warn(
            "project was quiescent with unresolved review findings; re-opening",
            findings=len(blocking),
            titles=[record.title[:60] for record in blocking[:4]],
        )

    def _node_workspace(self, node: Node) -> tuple[Repo, Sandbox]:
        """Persistent branch and sandbox owned by one mutating node."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in node.id)
        branch = f"forge/node/{safe_id}"
        path = self.config.worktrees_dir / safe_id
        repo = self.repo.ensure_worktree(path, branch, base="HEAD")
        if repo.head() != self.repo.head() and not repo.is_dirty():
            # Pull newly integrated dependency work into a persistent attempt.
            integrated = self.repo.head()
            if not repo.merge(self.repo.branch(), message=f"sync main into {branch}"):
                # Discarding this result was the actual root cause of the
                # unbounded retry loop. A branch that will not take main is a
                # branch that cannot be integrated, so the attempt that follows
                # is guaranteed to raise MergeConflict, be retried, and arrive
                # back here to fail the same sync. Leaving the provisional work
                # "intact for a later repair" meant intact forever.
                #
                # Rebuild on the integrated head instead. The provisional
                # commits are unmergeable, so they were never going to reach
                # main; the node re-implements against the tree it must
                # actually merge into. Loud, because work is being dropped.
                log.warn(
                    "node branch cannot absorb main; rebuilding it on the integrated head",
                    node=node.id,
                    branch=branch,
                    discarded=repo.head(),
                    base=integrated,
                )
                self.ledger.emit(
                    EventType.WORKTREE_REBASED,
                    node_id=node.id,
                    branch=branch,
                    discarded=repo.head(),
                    base=integrated,
                )
                repo.reset_hard(integrated)
        sandbox = build_sandbox(self.config.sandbox, repo.path)
        toolchain = detect_toolchain(sandbox)
        ensure_dependencies(
            sandbox,
            toolchain,
            timeout=self.config.sandbox.install_timeout,
            enabled=self.config.sandbox.install_dependencies,
        )
        return repo, sandbox

    def _run_agent(
        self,
        node: Node,
        *,
        agent: Any | None = None,
        repo: Repo | None = None,
        sandbox: Sandbox | None = None,
    ) -> AgentResult:
        agent = agent or build_agent(node.kind)
        repo = repo or self.repo
        sandbox = sandbox or self.sandbox
        ctx = AgentContext(
            node=node,
            config=self.config,
            models=self.models,
            memory=self.memory,
            lessons=self.lessons,
            repo=repo,
            sandbox=sandbox,
            gates=self.gates,
            graph=self.graph,
            toolchain=self._toolchain,
            project=self._project_context(),
            artifacts_dir=self.config.artifacts_dir,
        )
        return agent.run(ctx)

    def _project_context(self) -> dict[str, Any]:
        """The stable, cacheable part of every prompt.

        Recomputed only when memory has changed, so repeated calls within a node
        produce a byte-identical prefix -- which is what makes provider-side
        prompt caching effective rather than theoretical.
        """
        assert self.project is not None
        # The revision is durable rather than process-local so guidance written
        # by ``forge tell`` invalidates a live daemon's cached prompt digest.
        version = self.memory.version
        if version != self._digest_version:
            from ..memory.context import summarize_records_for_digest

            records = []
            for kind in (MemoryKind.REQUIREMENT, MemoryKind.DECISION, MemoryKind.INTERFACE,
                         MemoryKind.CONVENTION, MemoryKind.ASSUMPTION):
                records.extend(self.memory.by_kind(kind, limit=25))
            self._digest = summarize_records_for_digest(records)
            self._digest_version = version
        return {
            "id": self.project.id,
            "name": self.project.name,
            "goal": self.project.goal,
            "digest": self._digest,
        }

    # ------------------------------------------------------------------
    # Applying results
    # ------------------------------------------------------------------

    def _commit_result(
        self,
        node: Node,
        result: AgentResult,
        lease: Any,
        *,
        repo: Repo | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        """Make an agent's output durable, in crash-safe order.

        Order matters and is chosen deliberately:

        1. **Git commit** -- the code. If we crash after this, the retry finds
           the work already committed and produces an empty diff, which is
           harmless.
        2. **Memory** -- what was learned. Safe to re-write; records are
           idempotent by title.
        3. **New nodes** -- the work discovered. Duplicates would be the one
           genuinely bad outcome, so this happens after the commit that would
           have made them unnecessary and before the node is marked done.
        4. **Node status** -- the last write. Until it lands, the node is still
           owned by this attempt and will simply be retried.
        """
        self.graph.verify_lease(lease)
        repo = repo or self.repo
        sandbox = sandbox or self.sandbox

        if not result.success:
            self._handle_failure(node, None, result=result, repo=repo)
            return

        # A successful agent may still have concluded that only a human can take
        # the next step -- the goal check that is not converging, for instance.
        # Its work is kept; the node is parked so the operator sees the question.
        if result.needs_human:
            if result.memory:
                self.memory.write_many(result.memory, node_id=node.id)
            self.graph.block(node.id, result.summary or "human input required", question=result.needs_human)
            self.stats.nodes_blocked += 1
            log.warn("node completed but needs human input", node=node.id, summary=result.summary[:120])
            return

        commit_sha = None
        if result.commit_message and (result.changed_files or repo.is_dirty()):
            commit_sha = repo.commit(result.commit_message, node_id=node.id)
        elif repo.is_dirty():
            # An agent that changed files without asking for a commit left the
            # tree dirty. Committing it anyway is safer than discarding work,
            # and the trailer records which node did it.
            commit_sha = repo.commit(
                f"chore: changes from {node.title[:80]}", node_id=node.id
            )

        # Mutating agents work on persistent branches. Their commit becomes real
        # only after it merges into main and the affected gates pass there.
        if repo.path != self.repo.path and commit_sha:
            commit_sha = self._integrate(node, repo, sandbox, result)
        if commit_sha:
            self.ledger.append(
                Event(
                    type=EventType.PATCH_APPLIED,
                    node_id=node.id,
                    payload={
                        "commit": commit_sha,
                        "files": result.changed_files[:50],
                        "message": result.commit_message,
                    },
                )
            )

        # A node that wrote a manifest changed what this project *is*: new
        # gates become runnable, and the dependencies it declares do not exist
        # until something fetches them. Re-detecting here means the very next
        # node validates against a real toolchain rather than an absent one.
        if _touches_manifest(result.changed_files):
            self._refresh_toolchain()

        if result.memory:
            self.memory.write_many(result.memory, node_id=node.id)

        # A node created to fix a review finding closes that finding when it
        # succeeds. `MemoryStore.resolve_finding` shipped with no callers at
        # all, so findings only ever accumulated: a real project reached 36
        # open findings, most of them long since fixed, and nothing downstream
        # could distinguish an outstanding defect from a stale record.
        finding_id = node.spec.get("resolves_finding")
        if finding_id:
            self.memory.resolve_finding(str(finding_id), node_id=node.id)

        # A scripted user flow describes the product, so it outlives the QA
        # node that authored it. Kept in the node's own spec it made the only
        # behavioural gate applicable exactly once. Agents do not write to the
        # ledger, so the durable copy is taken here.
        flow_steps = ((result.data or {}).get("flow") or {}).get("steps")
        if flow_steps:
            self.ledger.kv_set(SMOKE_FLOW_KEY, {"steps": flow_steps, "authored_by": node.id})

        created = self._add_nodes(node, result.nodes)

        payload = {
            **result.to_dict(),
            "commit": commit_sha,
            "created_nodes": [n.id for n in created],
        }

        # Feed the router one outcome for the completed attempt.  Schema
        # compliance is not task success; only the validated node result is.
        # Pairing this with the failure path keeps observations bounded by real
        # attempts and prevents a multi-call repair loop from voting repeatedly.
        served_by = self._last_model_for_attempt(node.id)
        if served_by is not None:
            self.models.policy.record(
                _task_class_for(node.kind),
                served_by,
                success=True,
                node_id=node.id,
            )
        self.graph.succeed(node.id, payload)
        self.stats.nodes_completed += 1

        self.checkpoints.create(
            f"{node.kind}: {node.title[:60]}",
            kind="node",
            node_id=node.id,
            milestone=node.milestone,
            metadata={"commit": commit_sha, "created_nodes": len(created)},
        )

        if result.milestone_reached:
            self._on_milestone(node, result.milestone_reached)

        log.info(
            "node succeeded",
            node=node.id,
            title=node.title[:70],
            summary=result.summary[:120],
            created_nodes=len(created),
            commit=(commit_sha or "")[:8],
        )

    def _add_nodes(self, parent: Node, proposals: list[ProposedNode]) -> list[Node]:
        if not proposals:
            return []
        specs: list[dict[str, Any]] = []
        for proposal in proposals:
            specs.append(
                {
                    "kind": proposal.kind,
                    "title": proposal.title,
                    "spec": proposal.spec,
                    "deps": proposal.deps,
                    "parent_id": parent.id,
                    "priority": proposal.priority,
                    "milestone": proposal.milestone or parent.milestone,
                    "actor": f"node:{parent.id}",
                }
            )
        return self.graph.add_many(specs)

    def _on_milestone(self, node: Node, milestone: str) -> None:
        """React to a completed milestone: checkpoint, then plan the next one.

        Replanning here rather than up front is what lets the plan for milestone
        three benefit from everything learned building milestones one and two.
        """
        self.ledger.emit(EventType.MILESTONE_REACHED, node_id=node.id, milestone=milestone)
        self.checkpoints.create(f"milestone: {milestone}", kind="milestone", milestone=milestone)
        self.memory.compact()
        self._harvest_escalations()

        next_milestone = self._next_milestone(milestone)
        if next_milestone is None:
            log.info("all planned milestones complete", milestone=milestone)
            return
        existing = [n for n in self.graph.all_nodes() if n.milestone == next_milestone]
        if existing:
            return
        self.graph.add_node(
            NodeKind.PLAN,
            f"Plan milestone '{next_milestone}'",
            spec={
                "objective": f"Plan the work for milestone '{next_milestone}'",
                "acceptance": [f"A task graph exists for '{next_milestone}'"],
                "milestone": next_milestone,
                "replan": True,
            },
            priority=2,
            milestone=next_milestone,
            actor="orchestrator",
        )
        log.info("queued planning for the next milestone", milestone=next_milestone)

    def _next_milestone(self, current: str) -> str | None:
        record = next(
            (r for r in self.memory.by_kind(MemoryKind.FACT, limit=200) if r.title == "Milestone plan"),
            None,
        )
        if record is None:
            return None
        names: list[str] = []
        for line in record.body.splitlines():
            if ". " in line and ":" in line:
                names.append(line.split(". ", 1)[1].split(":", 1)[0].strip())
        if current not in names:
            return None
        index = names.index(current)
        return names[index + 1] if index + 1 < len(names) else None

    def _integrate(
        self, node: Node, repo: Repo, sandbox: Sandbox, result: AgentResult
    ) -> str:
        """Merge an isolated result and revalidate the integrated tree."""
        with self._integration_lock:
            # Old Forge versions wrote dependency fingerprints inside the
            # repository. Persistent branches may still contain commits that
            # changed that operational file. Neutralise it against main before
            # merging so Forge's own metadata cannot conflict with source.
            if repo.match_paths(self.repo.head(), [STAMP_PATH]):
                repo.commit("chore: exclude Forge dependency metadata")
            before = self.repo.head()
            branch = repo.branch()
            merge_message = f"merge: {node.title[:72]}\n\nForge-Node: {node.id}"
            if not self.repo.merge(branch, message=merge_message):
                raise MergeConflict(
                    "isolated work conflicts with newer integrated work",
                    branch=branch,
                )
            # The branch may declare a dependency that existed in its own
            # worktree but not in the integrated workspace.  Gates must see
            # the dependency tree described by the commit they are validating;
            # installing only after validation makes a correct dependency
            # addition fail on main and sends the node into pointless repair.
            integrated_toolchain = detect_toolchain(self.sandbox)
            install_record = ensure_dependencies(
                self.sandbox,
                integrated_toolchain,
                timeout=self.config.sandbox.install_timeout,
                enabled=self.config.sandbox.install_dependencies,
            )
            if install_record is not None:
                self.ledger.emit(EventType.TOOLCHAIN_INSTALLED, **install_record)
                if install_record.get("ok"):
                    integrated_toolchain = detect_toolchain(self.sandbox)
            gate_ctx = self.gates.build_context(
                root=self.repo.path,
                sandbox=self.sandbox,
                toolchain=integrated_toolchain,
                node_id=node.id,
                changed_files=result.changed_files,
                settings=node.spec.get("gate_settings", {}),
                memory=self.memory,
            )
            report = self.gates.run(
                node.gates or list(self.config.validation.gates),
                gate_ctx,
                use_cache=False,
            )
            if not report.passed:
                self.repo.reset_hard(before)
                self._toolchain = detect_toolchain(self.sandbox)
                raise ValidationFailed(
                    "integrated result failed validation",
                    evidence=report.render()[:4000],
                )
            return self.repo.head()

    def _handle_failure(
        self,
        node: Node,
        error: Exception | None,
        result: AgentResult | None = None,
        *,
        repo: Repo | None = None,
    ) -> None:
        message = ""
        escalatable = False
        if result is not None:
            message = result.summary
            escalatable = result.needs_escalation
            if result.needs_human:
                self.graph.block(node.id, "agent requested human input", question=result.needs_human)
                self.stats.nodes_blocked += 1
                return
            if result.memory:
                # Even a failed attempt usually learned something worth keeping.
                self.memory.write_many(result.memory, node_id=node.id)
            if result.data.get("decomposed") and result.nodes:
                created = self._add_nodes(node, result.nodes)
                fresh_spec = dict(node.spec)
                fresh_spec["decomposed"] = True
                fresh_spec["decomposition_children"] = [child.id for child in created]
                self.graph.update(
                    node.id,
                    spec=fresh_spec,
                    deps=[*node.deps, *[child.id for child in created]],
                    status=str(NodeStatus.PENDING),
                    actor="decomposition",
                )
                attempt_repo = repo or self.repo
                if attempt_repo.path != self.repo.path:
                    # Children must start from validated main, not inherit the
                    # broad parent's failing provisional implementation.
                    attempt_repo.reset_hard(self.repo.head())
                else:
                    self._preserve_or_discard_attempt(node, result, attempt_repo)
                log.info(
                    "replaced broad task with local child nodes",
                    node=node.id,
                    children=len(created),
                )
                return
        elif error is not None:
            message = f"{type(error).__name__}: {error}"

        fresh = self.graph.get(node.id)
        fresh = self._record_failure_progress(fresh, message, result)
        plan = self.scheduler.plan_failure(fresh, error, escalatable=escalatable)
        self.scheduler.apply(fresh, plan, message or "unknown failure")

        self._preserve_or_discard_attempt(node, result, repo or self.repo)

        if plan.disposition == Disposition.BLOCK:
            self.stats.nodes_blocked += 1
        elif plan.disposition == Disposition.FAIL:
            self.stats.nodes_failed += 1

        # A deterministic verdict about the model's output is the honest signal
        # for the router. Record it so a rung that keeps failing this class
        # stops being chosen for it.
        #
        # Attribute it to the model that *actually served* the work, not to
        # `node.tier`. That field only changes on an explicit escalation, so a
        # first-attempt failure served by a frontier rung was being recorded
        # against `local` -- teaching the router that local fails at tasks local
        # never attempted. Found by inspecting routing_stats after a live run.
        #
        # Only when the model actually produced the outcome. A rate limit, a
        # dropped connection or a bug in Forge's own code says nothing about
        # whether that rung can do this task class -- it never got to try. An
        # hour of provider instability was permanently biasing the router away
        # from a rung on evidence it had not generated.
        blames_the_model = error is None or (
            isinstance(error, ForgeError) and not error.transient
        )
        if blames_the_model:
            served_by = self._last_model_for_attempt(node.id)
            if served_by is not None:
                self.models.policy.record(
                    _task_class_for(node.kind),
                    served_by,
                    success=False,
                    node_id=node.id,
                )

        log.warn(
            "node attempt failed",
            node=node.id,
            title=node.title[:60],
            disposition=str(plan.disposition),
            reason=plan.reason,
            retry_in=round(plan.delay, 1),
        )

    def _record_failure_progress(
        self, node: Node, message: str, result: AgentResult | None
    ) -> Node:
        """Persist a stable failure signature and the strategies already tried."""
        normalized = re.sub(r"\b\d+\b", "#", (message or "unknown").lower())
        signature = hashlib.sha256(normalized[:4000].encode()).hexdigest()[:16]
        spec = dict(node.spec)
        same = int(spec.get("_same_failure_count", 0)) + 1
        if spec.get("_failure_signature") != signature:
            same = 1
        strategies = set(spec.get("_strategies_tried", []))
        model = self.config.models.models.get(node.tier)
        strategies.add("cloud-solve" if model and model.hosted == "cloud" else node.tier)
        if result and any(record.title.startswith("Coach advice") for record in result.memory):
            strategies.add("coach")
        if spec.get("decomposed"):
            strategies.add("decompose")
        spec.update(
            {
                "_failure_signature": signature,
                "_same_failure_count": same,
                "_strategies_tried": sorted(strategies),
            }
        )
        self.graph.update(node.id, spec=spec, actor="failure-progress")
        return self.graph.get(node.id)

    def _preserve_or_discard_attempt(
        self, node: Node, result: AgentResult | None, repo: Repo
    ) -> None:
        """Put back what a failed attempt wrote, so the retry starts clean.

        The *retry* case is already covered:
        :meth:`CheckpointManager.restore_for_attempt` resets the tree at the top
        of every attempt. This closes the case that reset cannot reach -- the
        failure that blocks the node or fails it terminally, where there is no
        attempt N+1 to do the resetting. Those edits otherwise sit in the tree
        until some unrelated node's attempt resets it, and in the meantime the
        dirty-tree branch of :meth:`_commit_result` can sweep them into that
        node's commit. A blocked node holds no claim on a tree others keep
        building in, and what failed is already durable in the ledger and in the
        findings the failure path writes to memory.

        Cleaning up promptly also means the cleanup can be *narrow*. Scoped to
        this attempt's own files, unlike the whole-tree ``reset_hard`` that
        ``restore_for_attempt`` performs -- which, with more than one worker on
        one checkout, discards whatever the other worker has not yet committed.
        """
        if repo.path != self.repo.path:
            preserve = bool(result and result.data.get("preserve_progress"))
            if preserve and repo.is_dirty():
                repo.commit(
                    f"wip: preserve attempt {node.attempts} for {node.title[:60]}",
                    node_id=node.id,
                )
                log.info("preserved provisional node progress", node=node.id)
            elif repo.is_dirty():
                changed = list(result.changed_files) if result else []
                repo.restore_paths(changed)
            return

        changed = list(result.changed_files) if result else []
        if not changed:
            return
        try:
            restored = self.repo.restore_paths(changed)
        except GitError as exc:  # pragma: no cover - git itself failing
            log.warn("could not discard the failed attempt", node=node.id, error=str(exc))
            return
        if restored:
            log.info(
                "discarded a failed attempt's edits",
                node=node.id,
                files=len(restored),
            )

    def _last_model_for_attempt(self, node_id: str) -> str | None:
        """Which model actually returned output during the current attempt.

        A node can be started many times without making a fresh model call --
        for example after a daemon restart, an expired lease, or a failure in
        Forge itself.  Looking up the node's latest request without respecting
        the attempt boundary repeatedly blamed a stale frontier response for
        every such failure.  One live node accumulated 317 supposed Opus
        failures from only a handful of real calls, poisoning both adaptive
        routing and the retrospective.

        Read the boundary from the durable ledger so this remains correct
        across processes and restarts.  A response (or validated cache hit),
        rather than a request, proves that the model produced something this
        attempt and is therefore eligible to receive the gate verdict.
        """
        events = self.ledger.read(
            node_id=node_id,
            types=[EventType.NODE_STARTED, EventType.MODEL_RESPONSE, EventType.MODEL_CACHE_HIT],
        )
        started_seq = max(
            (event.seq for event in events if event.type == EventType.NODE_STARTED),
            default=0,
        )
        for event in reversed(events):
            if event.seq <= started_seq:
                break
            if event.type in (EventType.MODEL_RESPONSE, EventType.MODEL_CACHE_HIT):
                model = event.payload.get("model")
                if isinstance(model, str) and model:
                    return model
        return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        counts = self.graph.counts()
        blocked = [
            {"id": n.id, "title": n.title, "question": (n.result or {}).get("question", "")}
            for n in self.graph.all_nodes(status=NodeStatus.BLOCKED)
        ]
        return {
            "project": self.project.to_dict() if self.project else None,
            "counts": counts,
            "progress": round(self.graph.progress(), 4),
            "quiescent": self.graph.is_quiescent(),
            "stalled": self.scheduler.stalled(),
            "critical_path": [
                {"id": n.id, "title": n.title, "status": n.status} for n in self.graph.critical_path()[:10]
            ],
            "blocked": blocked,
            "budget": self.models.budget.report(),
            "cache": self.models.cache.stats(),
            "memory": self.memory.counts(),
            "lessons": self.lessons.stats(),
            "stats": self.stats.to_dict(),
            "toolchain": self._toolchain,
            "stub_runs": self._stub_run_count(),
            "quiet_for": self._quiet_for(),
            "quiet_threshold": self._quiet_threshold(),
            "warning": self._latest_warning(),
        }

    def _latest_warning(self) -> dict[str, Any] | None:
        """The most recent unresolved environmental warning, if any.

        Only warnings newer than the last usage report matter: an endpoint that
        answered since is no longer the explanation for anything.
        """
        events = self.ledger.tail(1, types=[EventType.RUN_WARNING])
        if not events:
            return None
        payload = events[-1].payload
        if payload.get("resolved"):
            return None
        return {"kind": payload.get("kind", ""), "detail": payload.get("detail", "")}

    def _quiet_threshold(self) -> float:
        """How long silence has to last before it means something is wrong.

        Derived from the slowest rung rather than fixed, because "too quiet" is
        entirely relative to how long one call may legitimately take. A 15-minute
        constant was fine at 85 tok/s and became a false alarm on every healthy
        `local_deep` call once the local model dropped to 12 tok/s and its
        timeout went to an hour. A warning that fires during normal operation is
        worse than no warning: it trains the operator to ignore it.
        """
        timeouts = [
            self.models.registry.spec(name).timeout for name in self.config.models.ladder
        ]
        slowest = max(timeouts, default=900.0)
        # One full call plus half again, so an ordinary call plus its gate run
        # never trips it.
        return slowest * 1.5

    #: Events that mean work actually moved. Heartbeats, lease renewals and usage
    #: reports are deliberately excluded: they continue at full rate while a run
    #: makes no progress at all, which is exactly the state worth surfacing.
    PROGRESS_EVENTS = (
        EventType.NODE_STARTED,
        EventType.NODE_SUCCEEDED,
        EventType.NODE_FAILED,
        EventType.NODE_BLOCKED,
        EventType.MODEL_RESPONSE,
        EventType.GATE_PASSED,
        EventType.GATE_FAILED,
        EventType.CHECKPOINT_CREATED,
    )

    def _quiet_for(self) -> float:
        """Seconds since anything actually happened.

        "Working" in the status block means the graph has unfinished nodes, which
        stays true while a worker is wedged on a call that will never return. A
        run once sat for 45 minutes with a full heartbeat and no progress, and
        nothing in the status output distinguished that from healthy work.
        """
        events = self.ledger.tail(1, types=list(self.PROGRESS_EVENTS))
        if not events:
            return 0.0
        return max(0.0, self._clock.now() - events[-1].ts)

    def _stub_run_count(self) -> int:
        """How many of this project's runs were rehearsals on the echo stub.

        Reported because a dry run advances the graph exactly like a real one:
        nodes succeed, checkpoints land, progress climbs. Without this, a
        project whose entire history is a rehearsal reads as nearly finished.
        """
        return sum(
            1
            for event in self.ledger.read(types=[EventType.RUN_STARTED])
            if event.payload.get("dry_run")
        )

    def close(self) -> None:
        self.sandbox.teardown()
        self.ledger.close()

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _task_class_for(kind: str) -> str:
    from ..agents.registry import agent_registry, all_kinds

    all_kinds()
    try:
        return str(agent_registry.create(kind).task_class)
    except Exception:  # pragma: no cover
        return "unknown"


#: Files that declare what a project depends on. Writing one of these changes
#: which gates can run, so the toolchain is re-detected rather than kept from
#: the start of the run -- when the workspace was very likely empty.
_MANIFESTS = frozenset({
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "pyproject.toml", "requirements.txt", "poetry.lock",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
})


def _touches_manifest(paths: list[str]) -> bool:
    return any(Path(p).name in _MANIFESTS for p in paths or [])


def _slug(text: str, limit: int = 40) -> str:
    import re

    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:limit] or "project"
