#!/usr/bin/env python3
"""Adding a project-specific validation gate.

The most valuable thing you can do to improve an autonomous build is give it a
deterministic check for the mistake it keeps making. A gate costs nothing per run
and never forgets, whereas a model has to rediscover the problem every time --
which is exactly what Forge's own promotion detection is looking for when it
reports "this problem has been found 4 times".

This example adds a gate for a rule no off-the-shelf linter knows: every HTTP
handler in the project must have a timeout.

Run with:  python examples/custom-gate.py /path/to/project
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from forge.validation.gate import Gate, GateContext, register
from forge.validation.types import Issue, Severity, Verdict


@register
class HandlerTimeoutGate(Gate):
    """Every fetch/request call must specify a timeout."""

    name = "handler_timeout"
    description = "Outbound HTTP calls declare a timeout"

    # Cheap and text-only, so it runs early and its verdict is cacheable.
    order = 25
    # Advisory: report it, do not stall the run over it.
    blocking = False
    cacheable = True
    # Only these files can change the verdict, so edits elsewhere reuse the cache.
    inputs = ("*.ts", "*.tsx", "*.js", "*.py")

    CALLS = (
        (re.compile(r"\bfetch\s*\("), re.compile(r"\b(signal|AbortSignal|timeout)\b")),
        (re.compile(r"\brequests\.(get|post|put|delete)\s*\("), re.compile(r"\btimeout\s*=")),
        (re.compile(r"\bhttpx\.(get|post|Client)\s*\("), re.compile(r"\btimeout\s*=")),
    )

    def applicable(self, ctx: GateContext) -> bool:
        # Skip cleanly on projects this cannot apply to. A skip is recorded, so
        # the retrospective can notice a gate that never runs.
        return any(ctx.root.rglob(f"*{suffix}") for suffix in (".ts", ".js", ".py"))

    def run(self, ctx: GateContext) -> Verdict:
        issues: list[Issue] = []
        scanned = 0

        for path in sorted(ctx.root.rglob("*")):
            if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".jsx", ".py"}:
                continue
            rel = path.relative_to(ctx.root)
            if any(part in {"node_modules", ".git", "dist", "build", ".forge"} for part in rel.parts):
                continue

            scanned += 1
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for lineno, line in enumerate(lines, start=1):
                for call, guard in self.CALLS:
                    if not call.search(line):
                        continue
                    # Look at the call and the few lines after it, since options
                    # are usually on following lines.
                    window = "\n".join(lines[lineno - 1 : lineno + 4])
                    if not guard.search(window):
                        issues.append(
                            Issue(
                                message="outbound HTTP call without a timeout",
                                severity=Severity.MEDIUM,
                                path=rel.as_posix(),
                                line=lineno,
                                rule="handler-timeout",
                            )
                        )

        return Verdict(
            gate=self.name,
            passed=not issues,
            summary=f"{scanned} file(s) scanned, {len(issues)} call(s) without a timeout",
            issues=issues,
        )

    def version(self) -> str:
        # Bump when the logic changes, to invalidate previously cached verdicts.
        return "1"


def main(project_dir: Path) -> int:
    """Run the gate by hand against a project."""
    from forge.config import load_config
    from forge.kernel.ledger import Ledger
    from forge.validation.runner import GateRunner
    from forge.workspace.sandbox import build_sandbox, detect_toolchain

    config = load_config(project_dir)
    config.ensure_dirs()
    config.validation.gates = [*config.validation.gates, HandlerTimeoutGate.name]

    ledger = Ledger(config.ledger_path)
    try:
        sandbox = build_sandbox(config.sandbox, config.workspace_dir)
        runner = GateRunner(config, ledger)
        ctx = runner.build_context(
            root=config.workspace_dir,
            sandbox=sandbox,
            toolchain=detect_toolchain(sandbox),
            node_id="manual",
        )
        report = runner.run([HandlerTimeoutGate.name], ctx, use_cache=False)
        print(report.render(failures_only=False))
        return 0 if report.passed else 1
    finally:
        ledger.close()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    raise SystemExit(main(target))


# To use this permanently rather than by hand:
#
#   1. Put the module somewhere importable by the process running Forge.
#   2. Import it before the run (from a sitecustomize, or by adding it to
#      forge/validation/gates/__init__.py).
#   3. Add its name to `validation.gates` in .forge/config.toml.
#
# Alternatively, describe the rule as a *convention* in project memory. It costs
# nothing, reaches every agent's prompt automatically, and needs no code:
#
#   forge memory --kind convention
#
# Prefer prevention over detection, and detection over instruction: a lint rule
# that makes the mistake impossible beats a gate that catches it, which beats a
# sentence asking people not to make it. This gate is the middle option.
