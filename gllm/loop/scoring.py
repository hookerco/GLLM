from __future__ import annotations

import contextlib
import io
import re
from typing import Iterable

from gllm.loop.findings import Finding, Severity
from gllm.loop.types import Objective
from gllm.utils.plot_utils import parse_gcode

_SEVERITY_PENALTY = {Severity.ERROR: 1000.0, Severity.WARNING: 10.0, Severity.INFO: 0.1}
_TOOL_RE = re.compile(r"(?<![A-Z])T\s*([0-9]+)", re.IGNORECASE)


def findings_penalty(findings: Iterable[Finding]) -> float:
    """Lower is better; 0.0 means clean. Blocking findings dominate."""
    total = 0.0
    for f in findings:
        weight = _SEVERITY_PENALTY.get(f.severity, 1.0)
        total += weight if f.blocking else weight * 0.01
    return total


def path_length(gcode: str) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        xs, ys = parse_gcode(gcode)
    total = 0.0
    for i in range(1, len(xs)):
        total += ((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5
    return float(total)


def tool_changes(gcode: str) -> float:
    return float(sum(len(_TOOL_RE.findall(line)) for line in gcode.splitlines()))


_ESTIMATORS = {"path_length": path_length, "tool_changes": tool_changes}


def objective_value(gcode: str, objective: Objective | None) -> float | None:
    if objective is None:
        return None
    estimator = _ESTIMATORS[objective.metric]
    try:
        return estimator(gcode)
    except Exception:
        return None
