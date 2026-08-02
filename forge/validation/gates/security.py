"""Security gates.

Two checks that catch the two mistakes autonomous code generation actually
makes, as opposed to the ones security tooling usually looks for:

**Committed secrets.** A model asked to "wire up the API" will cheerfully write
a plausible key into a config file. The scanner is regex-based and tuned for low
false positives, because a noisy secret scanner gets ignored, and an ignored
secret scanner is worse than none.

**Vulnerable dependencies.** A model choosing a package picks the one it saw
most during training, which skews old. Running the ecosystem's own audit tool is
cheap and authoritative.

This is not a substitute for a security review of the finished product. It is
the deterministic floor beneath one, and the security *review agent* -- which
reads the actual code -- sits above it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...obs.log import get_logger
from ..gate import Gate, GateContext, register
from ..types import Issue, Severity, Verdict

log = get_logger("validation.security")

#: Patterns chosen for specificity, each with whether it is *heuristic*.
#:
#: The distinction matters. A structured credential -- an AWS key id, a GitHub
#: token -- is identified by a format that random text does not produce, so the
#: match is the evidence and second-guessing it with a word filter creates false
#: negatives (``AKIA...EXAMPLE`` contains "example" but so could a real key).
#: A heuristic match like ``password = "..."`` says nothing about the value, so
#: the placeholder filter is exactly right there.
SECRET_PATTERNS: tuple[tuple[str, str, bool], ...] = (
    ("aws-access-key", r"\bAKIA[0-9A-Z]{16}\b", False),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", False),
    ("slack-token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", False),
    ("google-api-key", r"\bAIza[0-9A-Za-z_\-]{35}\b", False),
    ("private-key", r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", False),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b", False),
    ("openai-key", r"\bsk-(?!ant)[A-Za-z0-9]{32,}\b", False),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", False),
    (
        "generic-assignment",
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|password|passwd|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]",
        True,
    ),
)

#: Values that look like secrets but are obviously not. Applied to heuristic
#: matches only.
PLACEHOLDER = re.compile(
    r"(?i)(your[_-]?|example|placeholder|dummy|sample|test|fake|xxx+|<[^>]+>|\.\.\.|changeme|replace[_-]?me)"
)

#: Credentials published in vendor documentation. Flagging these is noise, and
#: an explicit list is honest where a substring heuristic is not.
KNOWN_EXAMPLES = frozenset(
    {
        "AKIAIOSFODNN7EXAMPLE",
        "AKIAI44QH8DHBEXAMPLE",
        "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY",
    }
)


def _is_real_secret(value: str, heuristic: bool) -> bool:
    if value in KNOWN_EXAMPLES:
        return False
    if heuristic and PLACEHOLDER.search(value):
        return False
    return True

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".forge", "vendor"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
                 ".woff", ".woff2", ".ttf", ".mp4", ".webm", ".wasm", ".lock"}
SKIP_NAMES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock", "Cargo.lock"}


@register
class SecretScanGate(Gate):
    name = "secrets"
    description = "No credentials are committed to the repository"
    order = 15

    def run(self, ctx: GateContext) -> Verdict:
        compiled = [(name, re.compile(pattern), heuristic) for name, pattern, heuristic in SECRET_PATTERNS]
        issues: list[Issue] = []
        scanned = 0

        for path in sorted(ctx.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ctx.root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES or path.name in SKIP_NAMES:
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            scanned += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                if len(line) > 2000:
                    continue
                for name, pattern, heuristic in compiled:
                    match = pattern.search(line)
                    if not match:
                        continue
                    if not _is_real_secret(match.group(0), heuristic):
                        continue
                    # An env-var reference is the correct pattern, not a leak.
                    if "process.env" in line or "os.environ" in line or "getenv" in line:
                        continue
                    issues.append(
                        Issue(
                            message=f"possible {name} committed",
                            severity=Severity.CRITICAL,
                            path=rel.as_posix(),
                            line=lineno,
                            rule=name,
                        )
                    )
                    break

        return Verdict(
            gate=self.name,
            passed=not issues,
            summary=f"{scanned} file(s) scanned, {len(issues)} potential secret(s)",
            issues=issues,
        )


@register
class DependencyAuditGate(Gate):
    """Runs the ecosystem's dependency audit tool, if there is one."""

    name = "deps"
    description = "Dependencies have no known high-severity vulnerabilities"
    order = 90
    blocking = False  # a transitive advisory should not stall a whole project
    inputs = ("package.json", "package-lock.json", "pyproject.toml", "requirements.txt", "Cargo.toml")

    def applicable(self, ctx: GateContext) -> bool:
        return self._command(ctx) is not None

    def _command(self, ctx: GateContext) -> tuple[str, str] | None:
        if ctx.sandbox.exists("package-lock.json"):
            return "npm", "npm audit --json --audit-level=high || true"
        if ctx.sandbox.exists("pnpm-lock.yaml"):
            return "pnpm", "pnpm audit --json || true"
        if ctx.sandbox.exists("Cargo.toml"):
            return "cargo", "cargo audit --json || true"
        if ctx.sandbox.exists("requirements.txt") or ctx.sandbox.exists("pyproject.toml"):
            return "pip", "pip-audit -f json || true"
        return None

    def run(self, ctx: GateContext) -> Verdict:
        chosen = self._command(ctx)
        if chosen is None:
            return Verdict.skip(self.name, "no dependency manifest found")
        ecosystem, command = chosen
        result = ctx.sandbox.exec(command, shell=True, timeout=min(600.0, ctx.timeout))
        if "not found" in result.stderr.lower() or "command not found" in result.combined.lower():
            return Verdict.skip(self.name, f"{ecosystem} audit tool is not installed")

        issues = _parse_audit(ecosystem, result.stdout)
        blocking = [i for i in issues if i.severity == Severity.CRITICAL]
        return Verdict(
            gate=self.name,
            passed=not blocking,
            summary=f"{len(issues)} advisory/advisories from {ecosystem} audit",
            issues=issues[:30],
            evidence="" if not issues else result.stdout[:4000],
            detail={"ecosystem": ecosystem, "total": len(issues)},
        )


def _parse_audit(ecosystem: str, output: str) -> list[Issue]:
    try:
        data = json.loads(output or "{}")
    except json.JSONDecodeError:
        return []
    issues: list[Issue] = []
    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.CRITICAL,
        "moderate": Severity.MEDIUM,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
        "info": Severity.INFO,
    }

    if ecosystem in ("npm", "pnpm"):
        for name, entry in (data.get("vulnerabilities") or {}).items():
            level = str(entry.get("severity", "low")).lower()
            issues.append(
                Issue(
                    message=f"{name}: {level} severity advisory",
                    severity=severity_map.get(level, Severity.LOW),
                    rule="dependency",
                )
            )
    elif ecosystem == "pip":
        for entry in data.get("dependencies", data if isinstance(data, list) else []):
            for vuln in entry.get("vulns", []):
                issues.append(
                    Issue(
                        message=f"{entry.get('name')}: {vuln.get('id')}",
                        severity=Severity.MEDIUM,
                        rule="dependency",
                    )
                )
    elif ecosystem == "cargo":
        for entry in (data.get("vulnerabilities") or {}).get("list", []):
            advisory = entry.get("advisory", {})
            issues.append(
                Issue(
                    message=f"{advisory.get('package')}: {advisory.get('id')}",
                    severity=Severity.CRITICAL,
                    rule="dependency",
                )
            )
    return issues


@register
class DangerousPatternGate(Gate):
    """Flags a short list of patterns that are almost never right in new code.

    Kept short deliberately. A long list of "code smells" produces findings the
    system then spends tokens dismissing. These are the ones where the generated
    code is nearly always wrong to have written them.
    """

    name = "dangerous_patterns"
    description = "No obviously unsafe constructs in application code"
    order = 95
    blocking = False

    PATTERNS: tuple[tuple[str, str, str], ...] = (
        ("eval-string", r"\beval\s*\(", "eval on a runtime value"),
        ("child-process-shell", r"exec\s*\(\s*[`\"'].*\$\{", "shell command built by interpolation"),
        ("sql-interpolation", r"(?i)(?:SELECT|INSERT|UPDATE|DELETE)\s+.*\$\{", "SQL built by interpolation"),
        ("inner-html", r"\.innerHTML\s*=\s*(?!['\"]\s*['\"])", "innerHTML assigned a dynamic value"),
        ("disable-tls", r"rejectUnauthorized\s*:\s*false|verify\s*=\s*False", "TLS verification disabled"),
    )

    def run(self, ctx: GateContext) -> Verdict:
        compiled = [(name, re.compile(pattern), why) for name, pattern, why in self.PATTERNS]
        issues: list[Issue] = []
        for path in sorted(ctx.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".js", ".ts", ".jsx", ".tsx", ".py", ".mjs"}:
                continue
            rel = path.relative_to(ctx.root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for name, pattern, why in compiled:
                    if pattern.search(line):
                        issues.append(
                            Issue(
                                message=why,
                                severity=Severity.HIGH,
                                path=rel.as_posix(),
                                line=lineno,
                                rule=name,
                            )
                        )
        return Verdict(
            gate=self.name,
            passed=True,  # advisory: reported as findings, does not block
            summary=f"{len(issues)} risky construct(s) found",
            issues=issues[:40],
        )


def scan_text_for_secrets(text: str) -> list[str]:  # pragma: no cover - helper for agents
    """Used before writing model output to disk, as a last line of defence."""
    found = []
    for name, pattern, heuristic in SECRET_PATTERNS:
        match = re.search(pattern, text)
        if match and _is_real_secret(match.group(0), heuristic):
            found.append(name)
    return found


def redact(text: str) -> str:
    """Strip anything secret-shaped before it reaches a log or a cloud model."""
    for _, pattern, _heuristic in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def default_settings(root: Path) -> dict[str, Any]:  # pragma: no cover
    return {"root": str(root)}
