from __future__ import annotations

import re
from dataclasses import dataclass

from gllm.vericut.registry import VericutSetup, normalize_mode, normalize_tool


@dataclass(frozen=True)
class StaticFinding:
    code: str
    severity: str
    message: str
    line_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class StaticCheckReport:
    findings: tuple[StaticFinding, ...]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
        }


_BLOCK_COMMENT_RE = re.compile(r"\([^)]*\)")
_MODE_RE = re.compile(r"(?<![A-Z])G\s*([0-9]{1,3}(?:\.[0-9]+)?)", re.IGNORECASE)
_TOOL_RE = re.compile(r"(?<![A-Z])T\s*([0-9]+)", re.IGNORECASE)
_RAPID_RE = re.compile(r"(?<![A-Z])G\s*0?0(?![0-9.])", re.IGNORECASE)
_Z_RE = re.compile(r"(?<![A-Z])Z\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))", re.IGNORECASE)
_FEED_RE = re.compile(r"(?<![A-Z])F\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))", re.IGNORECASE)
_SPINDLE_RE = re.compile(r"(?<![A-Z])S\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))", re.IGNORECASE)
_AXIS_RE = re.compile(
    r"(?<![A-Z])([XYZ])\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))",
    re.IGNORECASE,
)
_UNIT_MODES = {"inch": "G20", "metric": "G21"}


def check_gcode(gcode: str, setup: VericutSetup) -> StaticCheckReport:
    lines = tuple(_clean_line(line) for line in gcode.splitlines())
    findings: list[StaticFinding] = []

    observed_modes = _observed_modes(lines)
    for mode in sorted(setup.required_modes):
        if mode not in observed_modes:
            findings.append(
                StaticFinding(
                    code="missing_required_mode",
                    severity="error",
                    message=f"Required setup mode {mode} was not found in generated G-code.",
                )
            )

    findings.extend(_expected_units_findings(observed_modes, setup))
    findings.extend(_unsupported_tools(lines, setup))
    findings.extend(_rapid_below_safe_z(lines, setup))
    findings.extend(_feed_rate_findings(lines, setup))
    findings.extend(_spindle_speed_findings(lines, setup))
    findings.extend(_work_envelope_findings(lines, setup))

    return StaticCheckReport(tuple(findings))


def _clean_line(line: str) -> str:
    without_semicolon_comment = line.split(";", 1)[0]
    return _BLOCK_COMMENT_RE.sub("", without_semicolon_comment).upper()


def _observed_modes(lines: tuple[str, ...]) -> frozenset[str]:
    modes: set[str] = set()
    for line in lines:
        for match in _MODE_RE.finditer(line):
            modes.add(normalize_mode(f"G{match.group(1)}"))
    return frozenset(modes)


def _unsupported_tools(lines: tuple[str, ...], setup: VericutSetup) -> list[StaticFinding]:
    if not setup.allowed_tools:
        return []

    findings: list[StaticFinding] = []
    reported: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        for match in _TOOL_RE.finditer(line):
            tool = normalize_tool(match.group(1))
            if tool in setup.allowed_tools or tool in reported:
                continue
            reported.add(tool)
            findings.append(
                StaticFinding(
                    code="unsupported_tool",
                    severity="error",
                    message=f"Tool {tool} is not allowed by setup {setup.id}.",
                    line_number=line_number,
                )
            )
    return findings


def _expected_units_findings(
    observed_modes: frozenset[str], setup: VericutSetup
) -> list[StaticFinding]:
    if setup.expected_units is None:
        return []

    expected_mode = _UNIT_MODES[setup.expected_units]
    unexpected_mode = "G21" if expected_mode == "G20" else "G20"
    if expected_mode not in observed_modes:
        if unexpected_mode in observed_modes:
            return [
                StaticFinding(
                    code="unexpected_units",
                    severity="error",
                    message=(
                        f"Setup {setup.id} expects {setup.expected_units} units "
                        f"({expected_mode}), but generated G-code uses {unexpected_mode}."
                    ),
                )
            ]
        return [
            StaticFinding(
                code="missing_expected_units",
                severity="error",
                message=(
                    f"Setup {setup.id} expects an explicit {expected_mode} units mode."
                ),
            )
        ]

    if unexpected_mode in observed_modes:
        return [
            StaticFinding(
                code="unexpected_units",
                severity="error",
                message=(
                    f"Setup {setup.id} expects {setup.expected_units} units "
                    f"({expected_mode}), but generated G-code also uses {unexpected_mode}."
                ),
            )
        ]

    return []


def _rapid_below_safe_z(lines: tuple[str, ...], setup: VericutSetup) -> list[StaticFinding]:
    if setup.safe_z_min is None:
        return []

    findings: list[StaticFinding] = []
    for line_number, line in enumerate(lines, start=1):
        if not _RAPID_RE.search(line):
            continue
        z_match = _Z_RE.search(line)
        if z_match is None:
            continue
        z_value = float(z_match.group(1))
        if z_value < setup.safe_z_min:
            findings.append(
                StaticFinding(
                    code="rapid_below_safe_z",
                    severity="error",
                    message=(
                        f"Rapid move Z{z_value:g} is below setup safe Z "
                        f"{setup.safe_z_min:g}."
                    ),
                    line_number=line_number,
                )
            )
    return findings


def _feed_rate_findings(lines: tuple[str, ...], setup: VericutSetup) -> list[StaticFinding]:
    return _range_findings(
        lines=lines,
        regex=_FEED_RE,
        minimum=setup.feed_rate_min,
        maximum=setup.feed_rate_max,
        code="feed_rate_out_of_range",
        label="Feed rate",
    )


def _spindle_speed_findings(lines: tuple[str, ...], setup: VericutSetup) -> list[StaticFinding]:
    return _range_findings(
        lines=lines,
        regex=_SPINDLE_RE,
        minimum=setup.spindle_speed_min,
        maximum=setup.spindle_speed_max,
        code="spindle_speed_out_of_range",
        label="Spindle speed",
    )


def _range_findings(
    lines: tuple[str, ...],
    regex: re.Pattern[str],
    minimum: float | None,
    maximum: float | None,
    code: str,
    label: str,
) -> list[StaticFinding]:
    if minimum is None and maximum is None:
        return []

    findings: list[StaticFinding] = []
    for line_number, line in enumerate(lines, start=1):
        for match in regex.finditer(line):
            value = float(match.group(1))
            if _within_range(value, minimum, maximum):
                continue
            findings.append(
                StaticFinding(
                    code=code,
                    severity="error",
                    message=(
                        f"{label} {value:g} is outside setup range "
                        f"{_format_range(minimum, maximum)}."
                    ),
                    line_number=line_number,
                )
            )
    return findings


def _work_envelope_findings(lines: tuple[str, ...], setup: VericutSetup) -> list[StaticFinding]:
    if setup.work_envelope is None:
        return []

    findings: list[StaticFinding] = []
    for line_number, line in enumerate(lines, start=1):
        for match in _AXIS_RE.finditer(line):
            axis = match.group(1).upper()
            value = float(match.group(2))
            minimum, maximum = setup.work_envelope.bounds_for(axis)
            if _within_range(value, minimum, maximum):
                continue
            findings.append(
                StaticFinding(
                    code="axis_out_of_envelope",
                    severity="error",
                    message=(
                        f"Axis {axis}{value:g} is outside setup envelope "
                        f"{_format_range(minimum, maximum)}."
                    ),
                    line_number=line_number,
                )
            )
    return findings


def _within_range(
    value: float, minimum: float | None, maximum: float | None
) -> bool:
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _format_range(minimum: float | None, maximum: float | None) -> str:
    lower = "-inf" if minimum is None else f"{minimum:g}"
    upper = "inf" if maximum is None else f"{maximum:g}"
    return f"[{lower}, {upper}]"
