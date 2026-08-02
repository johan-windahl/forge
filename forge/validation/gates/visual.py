"""Visual regression, decided by pixels.

"Does it look right?" is the question autonomous UI work most needs answered and
is least able to answer for itself. Forge splits it in two:

* **Has it changed?** -- a pixel comparison against an approved baseline. Cheap,
  exact, and reproducible. This gate.
* **Is it good?** -- a model looking at the screenshot. Expensive, subjective,
  and handled by the visual-review *agent*, which runs only when this gate says
  something changed.

That split is the whole efficiency argument. On a run where ninety UI screenshots
are captured and three differ, only three go to a vision model.

Comparison uses ImageMagick, which the target host already has and which is
substantially faster and more memory-stable over a multi-day run than decoding
PNGs in Python. If it is absent, the gate degrades to an exact-bytes comparison,
which is still a correct (if brittle) regression check.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ...obs.log import get_logger
from ...util.hashing import file_hash
from ...util.proc import run, which
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict

log = get_logger("validation.visual")

BASELINE_DIR = "visual-baselines"


def imagemagick_available() -> bool:
    return which("compare") is not None or which("magick") is not None


def compare_images(baseline: Path, candidate: Path, diff_out: Path) -> float | None:
    """Fraction of differing pixels in [0, 1], or ``None`` if incomparable.

    Uses the AE (absolute error) metric: the count of pixels that differ at all.
    Deliberately not a perceptual metric -- perceptual scores are tunable in a
    way that invites arguing with the number, whereas "how many pixels moved" is
    a fact. Anti-aliasing noise is handled by a small fuzz factor instead.
    """
    if not baseline.exists() or not candidate.exists():
        return None

    binary = which("compare")
    argv = (
        [binary, "-metric", "AE", "-fuzz", "2%", str(baseline), str(candidate), str(diff_out)]
        if binary
        else ["magick", "compare", "-metric", "AE", "-fuzz", "2%", str(baseline), str(candidate), str(diff_out)]
    )
    if not imagemagick_available():
        return 0.0 if file_hash(baseline) == file_hash(candidate) else 1.0

    diff_out.parent.mkdir(parents=True, exist_ok=True)
    result = run(argv, timeout=120, check=False)
    # `compare` writes the metric to stderr and exits non-zero when images
    # differ, which is not an error condition for us.
    match = re.search(r"(\d+(?:\.\d+)?)(?:e\+?(\d+))?", result.stderr.strip())
    if not match:
        if "image widths or heights differ" in result.stderr:
            return 1.0
        return None
    differing = float(match.group(1))
    if match.group(2):
        differing *= 10 ** float(match.group(2))

    total = _pixel_count(baseline)
    if not total:
        return None
    return min(1.0, differing / total)


def _pixel_count(path: Path) -> int:
    binary = which("identify") or which("magick")
    if binary is None:
        return 0
    argv = [binary, "-format", "%w %h", str(path)] if "identify" in binary else [binary, "identify", "-format", "%w %h", str(path)]
    result = run(argv, timeout=30, check=False)
    parts = result.stdout.split()
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]) * int(parts[1])
    return 0


@register
class VisualRegressionGate(Gate):
    """Compares captured screenshots against approved baselines."""

    name = "visual"
    description = "Screenshots match their approved baselines"
    order = 130
    cacheable = False
    # Visual change is usually *intended*. This gate reports and gates the
    # review, but a difference alone should not fail a node -- otherwise every
    # deliberate UI improvement stalls the run.
    blocking = False

    def applicable(self, ctx: GateContext) -> bool:
        return bool(list(_screenshots(ctx)))

    def run(self, ctx: GateContext) -> Verdict:
        tolerance = float(ctx.setting("tolerance", 0.005))
        baselines = ctx.root / BASELINE_DIR
        baselines.mkdir(parents=True, exist_ok=True)

        issues: list[Issue] = []
        artifacts: list[str] = []
        changed: list[dict[str, Any]] = []
        adopted = 0
        worst = 0.0

        for shot in _screenshots(ctx):
            baseline = baselines / shot.name
            if not baseline.exists():
                # First sighting: adopt as the baseline. Adoption is recorded so
                # a human reviewing the run can see which baselines were never
                # actually approved by anyone.
                shutil.copy2(shot, baseline)
                adopted += 1
                continue
            diff_path = ctx.artifact_path(f"diff_{shot.name}")
            fraction = compare_images(baseline, shot, diff_path)
            if fraction is None:
                issues.append(
                    Issue(message=f"could not compare {shot.name}", severity=Severity.LOW, rule="visual")
                )
                continue
            worst = max(worst, fraction)
            if fraction > tolerance:
                changed.append({"image": shot.name, "difference": round(fraction, 5)})
                artifacts.append(str(diff_path))
                issues.append(
                    Issue(
                        message=f"{shot.name} differs from baseline by {fraction:.2%}",
                        severity=Severity.MEDIUM,
                        path=shot.name,
                        rule="visual-diff",
                    )
                )

        summary_parts = []
        if adopted:
            summary_parts.append(f"{adopted} new baseline(s) adopted")
        if changed:
            summary_parts.append(f"{len(changed)} image(s) changed beyond {tolerance:.2%}")
        if not summary_parts:
            summary_parts.append("no visual change")

        return Verdict(
            gate=self.name,
            # Passing here means "no unexplained regression". Changes are
            # surfaced through detail so the review agent can pick them up.
            passed=True,
            summary="; ".join(summary_parts),
            issues=issues,
            artifacts=artifacts,
            score=round(worst, 5),
            detail={"changed": changed, "adopted": adopted, "tolerance": tolerance},
        )


def _screenshots(ctx: GateContext) -> list[Path]:
    directory = ctx.artifacts_dir / (ctx.node_id or "shared")
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.png")
        if not path.name.startswith(("diff_", "smoke_"))
    )


def approve_baselines(ctx: GateContext) -> int:
    """Promote current screenshots to baselines.

    Called after a visual review agent judges a change acceptable, and by
    ``forge approve`` when a human does. Keeping approval explicit is what stops
    a slow visual drift from going unnoticed over weeks of autonomous work.
    """
    baselines = ctx.root / BASELINE_DIR
    baselines.mkdir(parents=True, exist_ok=True)
    count = 0
    for shot in _screenshots(ctx):
        shutil.copy2(shot, baselines / shot.name)
        count += 1
    log.info("visual baselines approved", count=count)
    return count
