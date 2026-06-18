from __future__ import annotations

from typing import Protocol, runtime_checkable

from gllm.loop.findings import Finding, Severity, from_lint, from_static
from gllm.loop.types import ValidationContext
from gllm.utils import gcode_utils
from gllm.vericut.static_checks import check_gcode


@runtime_checkable
class Validator(Protocol):
    name: str
    tier: str

    def validate(self, gcode: str, ctx: ValidationContext) -> list[Finding]: ...


class LintValidator:
    """Wraps gcode_utils validators. Reliable checks block; noisy ones warn only."""

    name = "lint"
    tier = "lint"

    def validate(self, gcode: str, ctx: ValidationContext) -> list[Finding]:
        # (code, callable -> (ok, msg), blocking, severity)
        checks = [
            ("syntax", lambda: gcode_utils.validate_syntax(gcode), True, Severity.ERROR),
            ("unreachable_code", lambda: gcode_utils.validate_unreachable_code(gcode), True, Severity.ERROR),
            ("safety", lambda: gcode_utils.validate_safety(gcode), True, Severity.ERROR),
            ("feed_rate", lambda: gcode_utils.validate_feed_rate(gcode, ctx.feed_rate_min, ctx.feed_rate_max), True, Severity.ERROR),
            ("tool_changes", lambda: gcode_utils.validate_tool_changes(gcode), True, Severity.ERROR),
            ("spindle_speed", lambda: gcode_utils.validate_spindle_speed(gcode, ctx.spindle_speed_max), True, Severity.ERROR),
            ("tool_offsets", lambda: gcode_utils.check_tool_offsets(gcode), True, Severity.ERROR),
            # Known-noisy: non-blocking until logic is fixed (spec Section 8).
            ("continuity", lambda: gcode_utils.validate_continuity(gcode), False, Severity.WARNING),
        ]
        findings: list[Finding] = []
        for code, fn, blocking, severity in checks:
            try:
                result = fn()
            except Exception as exc:  # a validator crash is itself a (non-blocking) signal
                findings.append(
                    Finding("lint", f"{code}_error", Severity.WARNING, f"validator raised: {exc}", blocking=False)
                )
                continue
            finding = from_lint(code, result, blocking=blocking, severity=severity)
            if finding is not None:
                findings.append(finding)
        return findings


class StaticPolicyValidator:
    """Wraps gllm.vericut.static_checks.check_gcode (machine-policy gate)."""

    name = "static_policy"
    tier = "static"

    def validate(self, gcode: str, ctx: ValidationContext) -> list[Finding]:
        if ctx.setup is None:
            return []
        report = check_gcode(gcode, ctx.setup)
        return [from_static(f) for f in report.findings]
