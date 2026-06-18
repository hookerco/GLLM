from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
_SOURCE_RANK = {"static": 0, "vericut": 0, "lint": 1}
_VALID_SEVERITIES = {s.value for s in Severity}


@dataclass(frozen=True)
class Finding:
    source: Literal["lint", "static", "vericut"]
    code: str
    severity: Severity
    message: str
    line_number: int | None = None
    suggested_fix: str | None = None
    blocking: bool = True
    raw: dict | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "line_number": self.line_number,
            "suggested_fix": self.suggested_fix,
            "blocking": self.blocking,
        }


def _coerce_severity(value: object, default: Severity = Severity.ERROR) -> Severity:
    return Severity(value) if value in _VALID_SEVERITIES else default


def from_lint(
    name: str,
    result: tuple[bool, object],
    *,
    blocking: bool = True,
    severity: Severity = Severity.ERROR,
) -> Finding | None:
    ok, message = result
    if ok:
        return None
    return Finding(
        source="lint",
        code=name,
        severity=severity,
        message=str(message) if message else name,
        blocking=blocking,
    )


def from_static(finding) -> Finding:  # finding: gllm.vericut.static_checks.StaticFinding
    severity = _coerce_severity(finding.severity)
    return Finding(
        source="static",
        code=finding.code,
        severity=severity,
        message=finding.message,
        line_number=finding.line_number,
        blocking=(severity == Severity.ERROR),
        raw=finding.to_dict(),
    )


def from_vericut(payload: dict) -> Finding:
    severity = _coerce_severity(payload.get("severity"))
    return Finding(
        source="vericut",
        code=str(payload.get("code", "vericut_finding")),
        severity=severity,
        message=str(payload.get("message", "")),
        line_number=payload.get("line_number"),
        blocking=(severity == Severity.ERROR),
        raw=payload,
    )


def prioritize(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_RANK.get(f.severity, 3),
            _SOURCE_RANK.get(f.source, 2),
            f.line_number if f.line_number is not None else 1_000_000,
        ),
    )


def format_findings(findings: Iterable[Finding]) -> str:
    lines = []
    for f in prioritize(findings):
        loc = f" line {f.line_number}" if f.line_number is not None else ""
        lines.append(f"- [{f.severity.value}] {f.code}{loc}: {f.message}")
    return "\n".join(lines) if lines else "- No concrete findings were reported."
