"""Escalation pairs: the corpus for teaching the local model.

The value of a pair is that two attempts differ only in the model. These tests
hold the extractor to that, and to the harder requirement: a pair with no
evidence of *why* the weak attempt was rejected teaches nothing and must not
enter the corpus, because a corpus of unusable examples is worse than an empty
one -- it reads as data.
"""

from __future__ import annotations

from pathlib import Path

from forge.improve.escalation import EscalationCorpus, find_pairs, render
from forge.kernel.events import Event, EventType
from forge.kernel.ledger import Ledger

LADDER = ["local", "local_deep", "codex", "claude"]


def _response(ledger: Ledger, node: str, model: str, text: str, task_class: str = "implementation") -> None:
    ledger.append(Event(
        type=EventType.MODEL_RESPONSE,
        node_id=node,
        payload={"model": model, "tier": "local" if model.startswith("local") else "frontier",
                 "task_class": task_class, "text": text},
    ))


def _route(ledger: Ledger, node: str, model: str, outcome: str) -> None:
    ledger.append(Event(
        type=EventType.ROUTE_DECIDED,
        node_id=node,
        payload={"task_class": "implementation", "tier": model, "outcome": outcome},
    ))


def _gate_failed(ledger: Ledger, node: str, gate: str, summary: str) -> None:
    ledger.append(Event(
        type=EventType.GATE_FAILED, node_id=node, payload={"gate": gate, "summary": summary}
    ))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_a_weak_failure_then_strong_success_is_a_pair(ledger: Ledger) -> None:
    _response(ledger, "n1", "local", "export const x: any = 1;")
    _route(ledger, "n1", "local", "failure")
    _gate_failed(ledger, "n1", "types", "error TS7006: implicitly has an 'any' type")
    _response(ledger, "n1", "codex", "export const x: number = 1;")
    _route(ledger, "n1", "codex", "success")

    pairs = find_pairs(ledger, ladder=LADDER, project="p")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.weak.model == "local" and pair.strong.model == "codex"
    assert "any" in pair.weak.text and "number" in pair.strong.text
    assert pair.evidence and "TS7006" in pair.evidence[0]
    assert pair.diagnosable


def test_success_on_the_same_rung_is_not_an_escalation(ledger: Ledger) -> None:
    """A retry that eventually worked locally teaches nothing about the ladder."""
    _response(ledger, "n1", "local", "first try")
    _route(ledger, "n1", "local", "failure")
    _route(ledger, "n1", "local", "success")
    assert find_pairs(ledger, ladder=LADDER) == []


def test_a_node_that_never_succeeded_is_not_a_pair(ledger: Ledger) -> None:
    """Without a working answer there is nothing to compare against."""
    _route(ledger, "n1", "local", "failure")
    _route(ledger, "n1", "codex", "failure")
    assert find_pairs(ledger, ladder=LADDER) == []


def test_the_weakest_failure_and_strongest_success_are_chosen(ledger: Ledger) -> None:
    """The widest gap is the most informative comparison."""
    _response(ledger, "n1", "local", "weak")
    _response(ledger, "n1", "local_deep", "mid")
    _response(ledger, "n1", "claude", "strong")
    _route(ledger, "n1", "local", "failure")
    _route(ledger, "n1", "local_deep", "failure")
    _route(ledger, "n1", "claude", "success")

    pair = find_pairs(ledger, ladder=LADDER)[0]
    assert pair.weak.model == "local" and pair.strong.model == "claude"


# --------------------------------------------------------------------------
# Keeping the corpus usable
# --------------------------------------------------------------------------


def test_a_pair_without_rejecting_evidence_is_not_diagnosable(ledger: Ledger) -> None:
    """It records that an escalation happened -- which the router already knows."""
    _response(ledger, "n1", "local", "some output")
    _route(ledger, "n1", "local", "failure")
    _route(ledger, "n1", "codex", "success")

    pair = find_pairs(ledger, ladder=LADDER)[0]
    assert not pair.diagnosable


def test_undiagnosable_pairs_stay_out_of_the_corpus(tmp_path: Path, ledger: Ledger) -> None:
    """A corpus of unusable examples is worse than an empty one: it reads as data."""
    _response(ledger, "n1", "local", "output with no recorded rejection")
    _route(ledger, "n1", "local", "failure")
    _route(ledger, "n1", "codex", "success")

    corpus = EscalationCorpus(tmp_path / "escalations")
    assert corpus.record(find_pairs(ledger, ladder=LADDER)) == 0
    assert corpus.all() == []


def test_the_corpus_accumulates_across_projects(tmp_path: Path, ledger: Ledger) -> None:
    """One project's mistake is an anecdote; three projects' is a model property."""
    corpus = EscalationCorpus(tmp_path / "escalations")
    for project in ("pinball", "invoices"):
        _response(ledger, f"n-{project}", "local", "bad")
        _route(ledger, f"n-{project}", "local", "failure")
        _gate_failed(ledger, f"n-{project}", "types", "same mistake")
        _response(ledger, f"n-{project}", "codex", "good")
        _route(ledger, f"n-{project}", "codex", "success")
        corpus.record(find_pairs(ledger, ladder=LADDER, project=project))

    stats = corpus.stats()
    assert stats["pairs"] >= 2
    assert stats["by_task_class"]["implementation"] >= 2


def test_recording_the_same_run_twice_does_not_duplicate(tmp_path: Path, ledger: Ledger) -> None:
    _response(ledger, "n1", "local", "bad")
    _route(ledger, "n1", "local", "failure")
    _gate_failed(ledger, "n1", "types", "boom")
    _response(ledger, "n1", "codex", "good")
    _route(ledger, "n1", "codex", "success")

    corpus = EscalationCorpus(tmp_path / "escalations")
    pairs = find_pairs(ledger, ladder=LADDER, project="p")
    assert corpus.record(pairs) == 1
    assert corpus.record(pairs) == 0
    assert len(corpus.all()) == 1


def test_render_is_quiet_when_there_is_nothing(tmp_path: Path) -> None:
    assert "No escalation pairs" in render([])


# --------------------------------------------------------------------------
# Retention: without the text there is no pair
# --------------------------------------------------------------------------


def test_model_output_is_retained_for_learnable_task_classes(config, ledger: Ledger, provider) -> None:
    """`keep_transcripts` existed in config and was read nowhere.

    The ledger recorded that local failed and codex succeeded but not what
    either produced -- which is the only part worth learning from.
    """
    from forge.models.client import ModelClient
    from forge.models.types import Message, TaskClass, TaskProfile

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.responses = ["export const x: number = 1;"]
    client.complete(
        [Message("user", "write it")],
        TaskProfile(task_class=TaskClass.IMPLEMENTATION),
        node_id="n1",
    )

    event = ledger.read(types=[EventType.MODEL_RESPONSE])[-1]
    assert event.payload.get("text") == "export const x: number = 1;"


def test_output_is_not_retained_for_classes_that_teach_nothing(config, ledger: Ledger, provider) -> None:
    """Planning output is project-specific prose: ledger cost, no transferable rule."""
    from forge.models.client import ModelClient
    from forge.models.types import Message, TaskClass, TaskProfile

    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.responses = ["a long project plan " * 50]
    client.complete(
        [Message("user", "plan it")], TaskProfile(task_class=TaskClass.PLANNING), node_id="n1"
    )

    assert "text" not in ledger.read(types=[EventType.MODEL_RESPONSE])[-1].payload


def test_retained_output_is_capped(config, ledger: Ledger, provider) -> None:
    """The ledger must stay a log, not become a transcript archive."""
    from forge.models.client import ModelClient
    from forge.models.types import Message, TaskClass, TaskProfile

    config.memory.transcript_max_chars = 100
    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.responses = ["x" * 5000]
    client.complete(
        [Message("user", "q")], TaskProfile(task_class=TaskClass.IMPLEMENTATION), node_id="n1"
    )

    payload = ledger.read(types=[EventType.MODEL_RESPONSE])[-1].payload
    assert len(payload["text"]) == 100
    assert payload["text_truncated"] is True


def test_retention_can_be_switched_off(config, ledger: Ledger, provider) -> None:
    from forge.models.client import ModelClient
    from forge.models.types import Message, TaskClass, TaskProfile

    config.memory.keep_transcripts = False
    client = ModelClient(config, ledger)
    for name in config.models.providers:
        client.registry.install(name, provider)
    provider.responses = ["secret"]
    client.complete(
        [Message("user", "q")], TaskProfile(task_class=TaskClass.IMPLEMENTATION), node_id="n1"
    )

    assert "text" not in ledger.read(types=[EventType.MODEL_RESPONSE])[-1].payload
