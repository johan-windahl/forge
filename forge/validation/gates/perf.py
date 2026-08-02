"""Performance gates.

Performance is a requirement that autonomous systems silently violate, because
nothing in the normal loop notices a page that got two seconds slower. Two
measurements are taken, both cheap enough to run on every milestone:

* **Load performance** in a real browser: navigation timing plus first paint.
* **Bundle size**, which is a leading indicator that correlates with load time
  and is measurable without booting anything.

Budgets are configured per project. Absent a budget the gate still records the
numbers -- a trend line with no threshold is still worth having, and the
retrospective can propose a threshold once it has a baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...obs.log import get_logger
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict
from .browser import ServedApp, playwright_available

log = get_logger("validation.perf")

#: Directories that hold shippable output, checked for size.
BUNDLE_DIRS = ("dist", "build", "out", "public/build", ".next/static")


@register
class BundleSizeGate(Gate):
    """Total size of built assets against a budget."""

    name = "bundle_size"
    description = "Built assets stay within the size budget"
    order = 140
    blocking = False

    def applicable(self, ctx: GateContext) -> bool:
        return any((ctx.root / d).is_dir() for d in BUNDLE_DIRS)

    def run(self, ctx: GateContext) -> Verdict:
        budget_kb = float(ctx.setting("budget_kb", 0))
        totals: dict[str, int] = {}
        largest: list[tuple[str, int]] = []

        for directory in BUNDLE_DIRS:
            base = ctx.root / directory
            if not base.is_dir():
                continue
            total = 0
            for path in base.rglob("*"):
                if path.is_file() and path.suffix not in (".map",):
                    size = path.stat().st_size
                    total += size
                    largest.append((str(path.relative_to(ctx.root)), size))
            totals[directory] = total

        if not totals:
            return Verdict.skip(self.name, "no build output found")

        total_kb = sum(totals.values()) / 1024
        largest.sort(key=lambda pair: -pair[1])
        issues: list[Issue] = []
        passed = True
        if budget_kb and total_kb > budget_kb:
            passed = False
            issues.append(
                Issue(
                    message=f"bundle is {total_kb:.0f} kB, budget is {budget_kb:.0f} kB",
                    severity=Severity.MEDIUM,
                    rule="bundle-budget",
                )
            )
            issues += [
                Issue(message=f"{name} is {size / 1024:.0f} kB", severity=Severity.LOW, path=name)
                for name, size in largest[:5]
            ]

        return Verdict(
            gate=self.name,
            passed=passed,
            summary=f"built assets total {total_kb:.0f} kB",
            score=round(total_kb, 1),
            issues=issues,
            detail={"by_directory": {k: round(v / 1024, 1) for k, v in totals.items()},
                    "largest": [{"path": n, "kb": round(s / 1024, 1)} for n, s in largest[:10]]},
        )


@register
class LoadPerfGate(Gate):
    """Navigation and paint timings from a real browser load."""

    name = "load_perf"
    description = "Page load timings stay within budget"
    order = 150
    cacheable = False
    blocking = False

    def applicable(self, ctx: GateContext) -> bool:
        return playwright_available() and bool(
            ctx.setting("base_url") or ctx.toolchain.get("commands", {}).get("serve")
        )

    def run(self, ctx: GateContext) -> Verdict:
        if not playwright_available():
            return Verdict.skip(self.name, "playwright is not installed")

        budget_ms = float(ctx.setting("budget_ms", 0))
        route = str(ctx.setting("route", "/"))
        runs = int(ctx.setting("runs", 3))
        samples: list[dict[str, float]] = []

        try:
            with ServedApp(ctx) as app:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                    )
                    try:
                        for _ in range(max(1, runs)):
                            samples.append(self._measure(browser, app.base_url + route))
                    finally:
                        browser.close()
        except (RuntimeError, TimeoutError) as exc:
            return Verdict.failing(self.name, "the application did not start", evidence=str(exc)[:2000])
        except Exception as exc:  # pragma: no cover
            return Verdict.error(self.name, f"performance gate could not run: {exc}")

        if not samples:
            return Verdict.error(self.name, "no timing samples collected")

        # Median, not mean: one slow cold start should not define the verdict.
        def median(key: str) -> float:
            values = sorted(s.get(key, 0.0) for s in samples)
            return values[len(values) // 2]

        load_ms = median("load")
        issues: list[Issue] = []
        passed = True
        if budget_ms and load_ms > budget_ms:
            passed = False
            issues.append(
                Issue(
                    message=f"load took {load_ms:.0f}ms, budget is {budget_ms:.0f}ms",
                    severity=Severity.MEDIUM,
                    rule="load-budget",
                )
            )

        return Verdict(
            gate=self.name,
            passed=passed,
            summary=f"median load {load_ms:.0f}ms over {len(samples)} run(s)",
            score=round(load_ms, 1),
            issues=issues,
            detail={
                "samples": samples,
                "median_load_ms": round(load_ms, 1),
                "median_first_paint_ms": round(median("first_paint"), 1),
                "median_dom_content_loaded_ms": round(median("dom_content_loaded"), 1),
            },
        )

    @staticmethod
    def _measure(browser: Any, url: str) -> dict[str, float]:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            page.goto(url, wait_until="load", timeout=60_000)
            timing = page.evaluate(
                """() => {
                    const nav = performance.getEntriesByType('navigation')[0] || {};
                    const paint = performance.getEntriesByType('paint') || [];
                    const fp = paint.find(p => p.name === 'first-contentful-paint');
                    return {
                        load: nav.loadEventEnd || 0,
                        dom_content_loaded: nav.domContentLoadedEventEnd || 0,
                        first_paint: fp ? fp.startTime : 0,
                        transfer_bytes: nav.transferSize || 0,
                    };
                }"""
            )
            return {k: float(v) for k, v in (timing or {}).items()}
        finally:
            page.close()


@register
class BenchmarkGate(Gate):
    """Runs a project-defined benchmark command and reads a number from it.

    The contract is deliberately minimal: the command must print a line of the
    form ``FORGE_METRIC <name> <value>``. Anything more elaborate -- parsing
    arbitrary benchmark formats -- would be guesswork that fails silently.
    """

    name = "benchmark"
    description = "Project benchmark stays within budget"
    order = 160
    blocking = False

    def applicable(self, ctx: GateContext) -> bool:
        return bool(ctx.setting("command"))

    def run(self, ctx: GateContext) -> Verdict:
        command = str(ctx.setting("command", ""))
        if not command:
            return Verdict.skip(self.name, "no benchmark command configured")
        result = ctx.sandbox.exec(command, shell=True, timeout=min(1800.0, ctx.timeout))
        metrics: dict[str, float] = {}
        for line in result.combined.splitlines():
            if line.startswith("FORGE_METRIC "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        metrics[parts[1]] = float(parts[2])
                    except ValueError:
                        continue

        budgets: dict[str, float] = dict(ctx.setting("budgets", {}))
        issues = [
            Issue(
                message=f"{name} is {value:g}, budget is {budgets[name]:g}",
                severity=Severity.MEDIUM,
                rule="benchmark",
            )
            for name, value in metrics.items()
            if name in budgets and value > budgets[name]
        ]
        return Verdict(
            gate=self.name,
            passed=result.ok and not issues,
            summary=f"{len(metrics)} metric(s) captured",
            evidence="" if result.ok else result.combined,
            issues=issues,
            detail={"metrics": metrics},
        )


def bundle_report(root: Path) -> dict[str, Any]:  # pragma: no cover - reporting helper
    sizes: dict[str, float] = {}
    for directory in BUNDLE_DIRS:
        base = root / directory
        if base.is_dir():
            sizes[directory] = sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) / 1024
    return sizes
