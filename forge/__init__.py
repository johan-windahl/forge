"""Forge - an autonomous software engineering platform.

Forge takes a single high-level project description and drives it to shipped
software: planning, architecture, implementation, deterministic validation,
visual verification, deployment, and continuous self-improvement -- unattended,
for days at a time, across crashes.

The public surface is intentionally small. Most users interact through the
``forge`` CLI (see :mod:`forge.cli`) or embed :class:`forge.kernel.orchestrator.Orchestrator`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
