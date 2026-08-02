"""Platform feedback: detecting defects in Forge itself.

Every detector here was written against a real incident, and the fixtures
reproduce the shape of that incident rather than an invented one. The point of
this module is to notice a class of problem that the lessons library and the
routing policy structurally cannot -- so the tests hold it to that standard.
"""

from __future__ import annotations

from pathlib import Path

from forge.improve.feedback import (
    FeedbackStore,
    Finding,
    collect,
    detect_dead_rungs,
    detect_discarded_local_work,
    detect_identical_failures,
    detect_useless_gates,
    error_signature,
    render,
)
from forge.kernel.events import Event, EventType
from forge.kernel.ledger import Ledger
from forge.util.clock import ManualClock


def _fail(ledger: Ledger, node_id: str, error: str, *, disposition: str = "retry") -> None:
    ledger.append(
        Event(
            type=EventType.NODE_FAILED,
            node_id=node_id,
            payload={"error": error, "result": {"disposition": disposition}},
        )
    )


def _route(ledger: Ledger, node_id: str, task_class: str, tier: str, outcome: str) -> None:
    ledger.append(
        Event(
            type=EventType.ROUTE_DECIDED,
            node_id=node_id,
            payload={"task_class": task_class, "tier": tier, "outcome": outcome},
        )
    )


# --------------------------------------------------------------------------
# Signature normalisation
# --------------------------------------------------------------------------


def test_the_same_bug_normalises_to_one_signature() -> None:
    """Ids and numbers differ between two instances of a single defect."""
    a = 'codex exited 1 (thread_id="019fae8d-d1e8-7361-97b5-4530ab848cb2") invalid_json_schema'
    b = 'codex exited 1 (thread_id="7c2b1a90-0000-4444-8888-abcdefabcdef") invalid_json_schema'
    assert error_signature(a) == error_signature(b)


def test_genuinely_different_failures_stay_apart() -> None:
    """Over-normalising would merge distinct bugs and hide one of them."""
    a = "AssertionError: expected the ball to bounce off the flipper"
    b = "ProviderUnavailable: codex exited with invalid_json_schema"
    assert error_signature(a) != error_signature(b)


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


def test_identical_repeated_failures_are_reported_as_a_loop(ledger: Ledger) -> None:
    """The codex incident: 67 byte-identical failures, all classified retryable."""
    for i in range(8):
        _fail(
            ledger,
            "node_a",
            f'codex exited 1 (thread_id="019fae8d-d1e8-7361-97b5-4530ab8480{i:02d}") '
            "invalid_json_schema: Missing 'content'",
        )

    findings = detect_identical_failures(ledger)
    assert len(findings) == 1
    assert findings[0].severity == "critical", "an all-retryable loop is the worst case"
    assert findings[0].occurrences == 8


def test_a_node_failing_differently_each_time_is_not_flagged(ledger: Ledger) -> None:
    """That is a hard task, not a broken platform. Flagging it would be noise."""
    for i in range(8):
        _fail(ledger, "node_a", f"AssertionError: test_case_{i} expected {i} got {i + 1}")
    assert not detect_identical_failures(ledger)


def test_a_rung_that_never_succeeds_is_flagged(ledger: Ledger) -> None:
    for _ in range(5):
        _route(ledger, "node_a", "implementation", "codex", "failure")
    findings = detect_dead_rungs(ledger)
    assert findings and findings[0].severity == "critical"
    assert "codex" in findings[0].title and "implementation" in findings[0].title


def test_a_rung_that_sometimes_succeeds_is_not_flagged(ledger: Ledger) -> None:
    for _ in range(5):
        _route(ledger, "node_a", "implementation", "codex", "failure")
    _route(ledger, "node_a", "implementation", "codex", "success")
    assert not detect_dead_rungs(ledger)


def test_a_gate_that_never_passes_is_flagged(ledger: Ledger) -> None:
    """The types gate: 8 runs, 0 passes, over an uninstalled compiler."""
    for _ in range(4):
        ledger.append(Event(
            type=EventType.GATE_FAILED,
            payload={"gate": "types", "summary": "`npx --no-install tsc --noEmit` failed with exit 1"},
        ))
    findings = detect_useless_gates(ledger)
    assert findings and "types" in findings[0].title
    assert "npx" in findings[0].evidence[0]


def test_a_gate_that_catches_real_problems_is_not_flagged(ledger: Ledger) -> None:
    for _ in range(4):
        ledger.append(Event(type=EventType.GATE_FAILED, payload={"gate": "unit", "summary": "2 failed"}))
    ledger.append(Event(type=EventType.GATE_PASSED, payload={"gate": "unit", "summary": "all green"}))
    assert not detect_useless_gates(ledger)


def test_local_success_redone_on_cloud_is_flagged(ledger: Ledger) -> None:
    _route(ledger, "node_a", "implementation", "local", "success")
    _route(ledger, "node_a", "implementation", "claude", "success")
    findings = detect_discarded_local_work(ledger)
    assert findings and findings[0].occurrences == 1


def test_failed_cloud_attempts_do_not_inflate_the_redo_count(ledger: Ledger) -> None:
    """Otherwise one retry loop is counted twice, in two different findings.

    A report that double-counts an incident stops being usable as evidence,
    which is the only thing this module is for.
    """
    _route(ledger, "node_a", "implementation", "local", "success")
    for _ in range(20):
        _route(ledger, "node_a", "implementation", "codex", "failure")
    _route(ledger, "node_a", "implementation", "claude", "success")

    findings = detect_discarded_local_work(ledger)
    assert findings[0].occurrences == 1, "only the one successful redo actually repeated the work"


def test_a_clean_project_produces_no_findings(ledger: Ledger) -> None:
    """The report must stay silent when nothing is wrong, or it gets ignored."""
    _route(ledger, "node_a", "implementation", "local", "success")
    ledger.append(Event(type=EventType.GATE_PASSED, payload={"gate": "unit", "summary": "ok"}))
    assert collect(ledger, cloud_target=0.18) == []
    assert render([]) == "No platform anomalies detected."


# --------------------------------------------------------------------------
# Cross-project store
# --------------------------------------------------------------------------


def test_findings_merge_across_projects_rather_than_duplicating(tmp_path: Path) -> None:
    clock = ManualClock()
    store = FeedbackStore(tmp_path / "feedback", clock=clock)
    make = lambda project: Finding(  # noqa: E731
        kind="dead_rung", severity="high", title="Rung 'codex' has never succeeded",
        detail="d", occurrences=5, projects=[project],
    )

    assert store.record([make("pinball")]) == (1, 0)
    assert store.record([make("invoices")]) == (0, 1)

    stored = store.all()
    assert len(stored) == 1, "one defect, not one per project"
    assert stored[0].occurrences == 10
    assert set(stored[0].projects) == {"pinball", "invoices"}


def test_a_defect_seen_in_two_projects_is_escalated(tmp_path: Path) -> None:
    """Reproducing across unrelated projects is what proves it is the platform."""
    store = FeedbackStore(tmp_path / "feedback", clock=ManualClock())
    for project in ("pinball", "invoices"):
        store.record([Finding(
            kind="gate_never_passes", severity="high", title="Gate 'types' never passes",
            detail="d", projects=[project],
        )])
    assert store.all()[0].severity == "critical"
