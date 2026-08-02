"""Browser gates: does the thing actually run?

For anything with a user interface this is the gate that matters most. Unit
tests confirm that functions behave; only a real browser confirms that the
application boots, renders, and does not throw. In practice a large share of
autonomous-coding failures are caught here and nowhere else -- a module that
compiles, passes its tests, and produces a blank white page.

What is collected is chosen to be *diagnostic without being verbose*: console
errors, uncaught exceptions, failed network requests, and a screenshot. Those
four signals identify almost every boot failure, and together they cost a few
hundred tokens rather than a full DOM dump.

Playwright is an optional dependency. Without it the gate skips loudly rather
than failing, so a headless server without browsers still runs everything else.
"""

from __future__ import annotations

import re
import socket
import time
from typing import Any
from urllib.parse import urljoin

from ...obs.log import get_logger
from ...util.proc import BackgroundProcess
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict

log = get_logger("validation.browser")

DEFAULT_PORT = 5173


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServedApp:
    """Starts the project's dev server and guarantees it is torn down.

    A leaked dev server is uniquely nasty in an unattended run: the next attempt
    connects to the *previous* build, gates pass against stale code, and the
    system concludes work is done when it is not. Hence the context manager, the
    process-group kill, and the explicit port check rather than a fixed sleep.
    """

    def __init__(self, ctx: GateContext, *, port: int | None = None) -> None:
        self.ctx = ctx
        self.port = port or int(ctx.setting("port", 0)) or _free_port()
        self.command = ctx.setting("serve_command") or ctx.toolchain.get("commands", {}).get("serve")
        self.process: BackgroundProcess | None = None
        self.external = False

    @property
    def base_url(self) -> str:
        configured = self.ctx.setting("base_url")
        if configured:
            return str(configured)
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServedApp:
        if self.ctx.setting("base_url"):
            self.external = True
            return self
        if not self.command:
            return self
        command = str(self.command)
        env = {"PORT": str(self.port), "FORGE_PORT": str(self.port), "BROWSER": "none", "CI": "1"}
        log_path = self.ctx.artifact_path("server.log")
        self.process = self.ctx.sandbox.background(
            ["/bin/sh", "-lc", command], env=env, log_path=log_path
        )
        deadline = time.monotonic() + float(self.ctx.setting("startup_timeout", 90))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"dev server exited immediately: {self.process.output[-1500:]}"
                )
            if _port_open("127.0.0.1", self.port):
                # Bound, but frameworks often bind before they can serve.
                time.sleep(1.0)
                return self
            time.sleep(0.4)
        output = self.process.output[-1500:]
        self.stop()
        raise TimeoutError(f"dev server did not listen on {self.port}: {output}")

    def stop(self) -> None:
        if self.process:
            self.process.stop()
            self.process = None

    def __exit__(self, *exc: object) -> None:
        self.stop()


@register
class BrowserGate(Gate):
    """Loads the app and asserts it renders without errors."""

    name = "browser"
    description = "The application loads in a real browser without errors"
    order = 110
    # A real server and a real renderer are involved; treating a past pass as
    # proof of a present pass would be wrong.
    cacheable = False

    def applicable(self, ctx: GateContext) -> bool:
        has_ui = bool(
            ctx.setting("base_url")
            or ctx.toolchain.get("commands", {}).get("serve")
            or ctx.sandbox.exists("index.html")
        )
        return has_ui and playwright_available()

    def run(self, ctx: GateContext) -> Verdict:
        if not playwright_available():
            return Verdict.skip(self.name, "playwright is not installed")

        routes: list[str] = list(ctx.setting("routes", ["/"]))
        settle_ms = int(ctx.setting("settle_ms", 2500))
        issues: list[Issue] = []
        artifacts: list[str] = []
        detail: dict[str, Any] = {"routes": {}}

        try:
            with ServedApp(ctx) as app:
                if not app.external and app.process is None and not ctx.setting("base_url"):
                    return Verdict.skip(self.name, "no serve command available")
                from playwright.sync_api import sync_playwright

                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=bool(ctx.setting("headless", True)),
                        # Required in most container environments; harmless
                        # elsewhere. Hard-won, and exactly the sort of fact that
                        # becomes a lesson if it is ever not.
                        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                    )
                    try:
                        for route in routes:
                            page_issues, shot = self._visit(
                                browser, app.base_url, route, ctx, settle_ms
                            )
                            issues.extend(page_issues)
                            if shot:
                                artifacts.append(shot)
                            detail["routes"][route] = len(page_issues)
                    finally:
                        browser.close()
        except (RuntimeError, TimeoutError) as exc:
            return Verdict.failing(
                self.name,
                "the application did not start",
                evidence=str(exc)[:4000],
                issues=[Issue(message=str(exc)[:400], severity=Severity.CRITICAL, rule="startup")],
            )
        except Exception as exc:  # pragma: no cover - playwright internals
            return Verdict.error(self.name, f"browser gate could not run: {exc}")

        blocking = [i for i in issues if i.severity in (Severity.HIGH, Severity.CRITICAL)]
        return Verdict(
            gate=self.name,
            passed=not blocking,
            summary=(
                f"{len(routes)} route(s) loaded cleanly"
                if not blocking
                else f"{len(blocking)} blocking browser error(s)"
            ),
            issues=issues,
            artifacts=artifacts,
            detail=detail,
        )

    def _visit(self, browser: Any, base_url: str, route: str, ctx: GateContext, settle_ms: int):
        issues: list[Issue] = []
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        page.on(
            "console",
            lambda msg: issues.append(
                Issue(
                    message=f"console.{msg.type}: {msg.text[:300]}",
                    severity=Severity.HIGH if msg.type == "error" else Severity.LOW,
                    path=route,
                    rule="console",
                )
            )
            if msg.type in ("error", "warning")
            else None,
        )
        page.on(
            "pageerror",
            lambda exc: issues.append(
                Issue(
                    message=f"uncaught exception: {str(exc)[:300]}",
                    severity=Severity.CRITICAL,
                    path=route,
                    rule="pageerror",
                )
            ),
        )
        page.on(
            "requestfailed",
            lambda request: issues.append(
                Issue(
                    message=f"request failed: {request.url[:200]} ({request.failure})",
                    # A failed request for a source file is fatal; a failed
                    # analytics beacon is noise.
                    severity=Severity.HIGH
                    if any(request.url.endswith(ext) for ext in (".js", ".css", ".mjs", ".ts", ".wasm"))
                    else Severity.LOW,
                    path=route,
                    rule="network",
                )
            ),
        )

        shot_path = None
        try:
            url = base_url.rstrip("/") + route
            response = page.goto(
                url, wait_until="domcontentloaded", timeout=ctx.browser_timeout * 1000
            )
            if response is not None and response.status >= 400:
                issues.append(
                    Issue(
                        message=f"HTTP {response.status} for {route}",
                        severity=Severity.CRITICAL,
                        path=route,
                        rule="http",
                    )
                )
            page.wait_for_timeout(settle_ms)

            # A page that renders nothing is the classic silent failure: no
            # errors, no content. Check for it explicitly.
            body_text = (page.inner_text("body") or "").strip() if page.query_selector("body") else ""
            canvas = page.query_selector("canvas")
            if len(body_text) < 2 and canvas is None:
                issues.append(
                    Issue(
                        message="page rendered no visible text and no canvas",
                        severity=Severity.CRITICAL,
                        path=route,
                        rule="blank",
                    )
                )

            target = ctx.artifact_path(f"screenshot{route.replace('/', '_') or '_root'}.png")
            page.screenshot(path=str(target), full_page=False)
            shot_path = str(target)
        finally:
            context.close()
        return issues, shot_path


@register
class SmokeFlowGate(Gate):
    """Runs a scripted interaction described in the node spec.

    Steps are declarative -- click, fill, press, wait, expect_text -- rather than
    generated code. A model that writes Playwright code writes *plausible*
    Playwright code; a model that fills in a fixed step vocabulary either
    produces a runnable flow or fails schema validation immediately. The
    vocabulary is small on purpose and grows only when a real project needs it.
    """

    name = "smoke"
    description = "A scripted user flow completes"
    order = 120
    cacheable = False

    def applicable(self, ctx: GateContext) -> bool:
        return bool(ctx.setting("steps")) and playwright_available()

    def run(self, ctx: GateContext) -> Verdict:
        steps: list[dict[str, Any]] = list(ctx.setting("steps", []))
        if not steps:
            return Verdict.skip(self.name, "no flow defined")
        if not playwright_available():
            return Verdict.skip(self.name, "playwright is not installed")

        issues: list[Issue] = []
        artifacts: list[str] = []
        completed = 0
        try:
            with ServedApp(ctx) as app:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                    )
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    try:
                        page.goto(app.base_url, wait_until="domcontentloaded")
                        for index, step in enumerate(steps):
                            try:
                                self._run_step(page, step)
                                completed += 1
                            except Exception as exc:
                                issues.append(
                                    Issue(
                                        message=f"step {index + 1} ({step.get('action')}) failed: {str(exc)[:300]}",
                                        severity=Severity.CRITICAL,
                                        rule="flow",
                                    )
                                )
                                target = ctx.artifact_path(f"smoke_failure_{index + 1}.png")
                                page.screenshot(path=str(target))
                                artifacts.append(str(target))
                                break
                        final = ctx.artifact_path("smoke_final.png")
                        page.screenshot(path=str(final))
                        artifacts.append(str(final))
                    finally:
                        browser.close()
        except (RuntimeError, TimeoutError) as exc:
            return Verdict.failing(self.name, "the application did not start", evidence=str(exc)[:3000])
        except Exception as exc:  # pragma: no cover
            return Verdict.error(self.name, f"smoke gate could not run: {exc}")

        return Verdict(
            gate=self.name,
            passed=not issues,
            summary=f"{completed}/{len(steps)} flow steps completed",
            issues=issues,
            artifacts=artifacts,
            score=completed / len(steps) if steps else 1.0,
        )

    @staticmethod
    def _run_step(page: Any, step: dict[str, Any]) -> None:
        action = step.get("action")
        value = step.get("value", "")
        selector = str(step.get("selector", ""))
        if not selector and action in {"click", "fill", "wait_for", "expect_selector"}:
            # Local models commonly put the selector in `value` even though the
            # schema names a selector field. Accept that harmless representation
            # error instead of filing an application repair for a harness typo.
            selector = str(value)
            if selector.lower().startswith("selector:"):
                selector = selector.split(":", 1)[1].strip()
        timeout = float(step.get("timeout", 10_000))
        match action:
            case "click":
                page.click(selector, timeout=timeout)
            case "fill":
                page.fill(selector, str(value), timeout=timeout)
            case "press":
                page.press(selector or "body", str(value), timeout=timeout)
            case "key_down":
                page.keyboard.down(str(value))
            case "key_up":
                page.keyboard.up(str(value))
            case "wait":
                duration = float(value or 1000)
                # Decimal values such as "0.5" are naturally emitted as seconds;
                # conventional integer values such as "1000" remain milliseconds.
                page.wait_for_timeout(duration * 1000 if duration <= 60 else duration)
            case "wait_for":
                page.wait_for_selector(selector, timeout=timeout)
            case "goto":
                target = str(value)
                if target.startswith("/"):
                    target = urljoin(page.url, target)
                page.goto(target, wait_until="domcontentloaded", timeout=timeout)
            case "reload":
                page.reload(wait_until="domcontentloaded", timeout=timeout)
            case "resize":
                match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", str(value))
                if match is None:
                    raise ValueError("resize value must be WIDTHxHEIGHT")
                page.set_viewport_size(
                    {"width": int(match.group(1)), "height": int(match.group(2))}
                )
            case "tab_away":
                duration = float(value or 1)
                duration_ms = duration * 1000 if duration <= 60 else duration
                other = page.context.new_page()
                try:
                    other.goto("about:blank")
                    other.bring_to_front()
                    page.wait_for_timeout(duration_ms)
                    page.bring_to_front()
                finally:
                    other.close()
            case "expect_fps":
                thresholds = [
                    float(item.strip())
                    for item in str(value or "60,30").split(",")
                ]
                minimum_fps = thresholds[0]
                maximum_frame_ms = thresholds[1] if len(thresholds) > 1 else 30.0
                duration_ms = int(step.get("duration", 2000))
                metrics = page.evaluate(
                    """duration => new Promise(resolve => {
                      const deltas = [];
                      let previous;
                      const start = performance.now();
                      function frame(now) {
                        if (previous !== undefined) deltas.push(now - previous);
                        previous = now;
                        if (now - start >= duration) {
                          const mean = deltas.reduce((a, b) => a + b, 0) / Math.max(1, deltas.length);
                          resolve({ fps: 1000 / mean, maxFrameMs: Math.max(0, ...deltas) });
                        } else requestAnimationFrame(frame);
                      }
                      requestAnimationFrame(frame);
                    })""",
                    duration_ms,
                )
                if float(metrics["fps"]) < minimum_fps:
                    raise AssertionError(
                        f"measured {metrics['fps']:.1f} FPS, expected at least {minimum_fps:.1f}"
                    )
                if float(metrics["maxFrameMs"]) > maximum_frame_ms:
                    raise AssertionError(
                        f"frame spike {metrics['maxFrameMs']:.1f}ms exceeds {maximum_frame_ms:.1f}ms"
                    )
            case "expect_text":
                page.wait_for_selector(f"text={value}", timeout=timeout)
            case "expect_selector":
                page.wait_for_selector(selector, timeout=timeout)
            case _:
                raise ValueError(f"unknown step action {action!r}")


#: Schema for a flow, so an agent can propose one and have it validated.
FLOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "click", "fill", "press", "key_down", "key_up",
                            "wait", "wait_for", "goto", "reload", "resize",
                            "tab_away", "expect_fps", "expect_text",
                            "expect_selector",
                        ],
                    },
                    "selector": {"type": "string", "description": "CSS selector"},
                    "value": {"type": "string", "description": "Text, key, URL or milliseconds"},
                    "timeout": {"type": "integer", "minimum": 100},
                    "duration": {"type": "integer", "minimum": 100},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}
