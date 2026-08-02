#!/usr/bin/env python3
"""Driving Forge from Python instead of the CLI.

Useful when Forge is a step inside a larger system: a CI job that builds a
prototype from a ticket, a service that queues projects, or a harness that runs
the same goal against several configurations to compare them.

Run with:  python examples/embedding-forge.py /tmp/my-build
"""

from __future__ import annotations

import sys
from pathlib import Path

from forge.config import load_config
from forge.kernel.orchestrator import Orchestrator
from forge.obs.log import setup_logging
from forge.report.progress import render_status

GOAL = "A command-line tool that watches a directory and reports file changes as JSON lines."


def main(project_dir: Path) -> int:
    # Configuration is layered; overrides here sit above config files but below
    # nothing, so this is the programmatic equivalent of CLI flags.
    config = load_config(
        project_dir,
        overrides={
            "scheduler": {"workers": 2},
            "budget": {"total_cost": 20.0, "cloud_fraction_target": 0.1},
            "validation": {"gates": ["schema", "secrets", "lint", "unit"]},
        },
    )
    config.ensure_dirs()
    setup_logging(path=config.log_path, console=True, console_level="info")

    with Orchestrator(config) as orchestrator:
        if orchestrator.project is None:
            orchestrator.create_project(GOAL, name="dirwatch")

        # Subscribe to the event stream for live progress. Subscribers are for
        # live concerns only -- anything that must survive a restart belongs in a
        # projection, because a subscriber that was not running never sees the
        # event.
        def on_event(event) -> None:
            if event.type in ("node.succeeded", "node.blocked", "milestone.reached"):
                print(f"  [{event.type}] {event.payload.get('summary', event.node_id)}")

        orchestrator.ledger.subscribe(on_event)

        # `run` returns when the graph is quiescent, a limit is hit, or it is
        # stopped. Calling it again resumes -- including after a crash.
        summary = orchestrator.run(install_signal_handlers=False)

        print()
        print(render_status(summary, verbose=True))

        # Blocked nodes are the only thing needing a human. In an unattended
        # pipeline, this is where you would raise a ticket rather than block.
        for blocked in summary["blocked"]:
            print(f"\nNEEDS INPUT: {blocked['title']}\n{blocked['question']}")

        # Anything computable is computed, so a caller can gate on real numbers.
        from forge.improve.metrics import compute_metrics

        metrics = compute_metrics(orchestrator.ledger, orchestrator.graph)
        print(f"\nCost {metrics.cost_total:.4f}, cloud share {metrics.cloud_fraction:.1%}, "
              f"rework {metrics.rework_ratio:.2f}")

        return 0 if not summary["stalled"] else 3


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "build"
    raise SystemExit(main(target))
