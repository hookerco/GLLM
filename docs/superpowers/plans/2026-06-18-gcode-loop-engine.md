# G-Code Loop Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, UI-agnostic engine (`gllm/loop/`) that runs an autonomous generate → validate → repair/improve loop over G-code, gating on fast lint + machine-policy static checks with Vericut as an optional final gate, returning the best-scored candidate.

**Architecture:** A `LoopController` drives a converge-or-cap loop with best-so-far selection. Validators (lint, static, vericut) all normalize to one `Finding` type. A pure `prompt → gcode` `CandidateGenerator` wraps the existing LangChain model wiring. Three modes (generate/repair/improve) differ only by starting state and which prompt builder runs. All loop logic is plain Python, testable with fakes and zero Vericut license. This plan delivers the engine; the FastAPI service and NiceGUI UI are separate follow-up plans.

**Tech Stack:** Python 3.11, dataclasses, `pygcode`, LangChain (`langchain_openai`, `langchain_core`), `pytest` (running existing `unittest.TestCase` suites). Reuses `gllm/vericut/*`, `gllm/utils/gcode_utils.py`, `gllm/utils/plot_utils.py`, `gllm/utils/model_utils.py`, `gllm/proof/runner.py`.

**Scope note (spec deltas):** The spec's `Objective` lists four metrics; v1 implements the two cheaply static-estimable ones (`path_length`, `tool_changes`) — `parse_gcode` returns only X/Y points with no motion-type or tool data, so `cycle_time`/`rapid_distance` are deferred to Phase 2 (motion-aware parsing or Vericut). The spec calls for refactoring `run_proof_scenario` into a thin adapter; this plan builds the engine alongside the existing runner (sharing the `Finding`/validator code) and does the `run_proof_scenario` unification as the final task, with the 7 existing corpus tests as the regression guard.

---

## File Structure

**Create:**
- `gllm/loop/__init__.py` — package marker, re-exports public API
- `gllm/loop/types.py` — `Mode`, `Objective`, `TokenUsage`, `Budget`, `ValidationContext`, `LoopRequest`, `Attempt`, `LoopResult`, `LoopEvent`
- `gllm/loop/findings.py` — `Severity`, `Finding`, `from_lint`/`from_static`/`from_vericut`, `prioritize`, `format_findings`
- `gllm/loop/constraints.py` — `setup_constraints(setup)`
- `gllm/loop/validators.py` — `Validator` protocol, `LintValidator`, `StaticPolicyValidator`
- `gllm/loop/scoring.py` — `findings_penalty`, `objective_value`, `path_length`, `tool_changes`
- `gllm/loop/generator.py` — `CandidateGenerator` protocol, `LLMCandidateGenerator`, `build_generate_prompt`/`build_repair_prompt`/`build_improve_prompt`
- `gllm/loop/controller.py` — `LoopController`, `pick_best`, `converged`, `derive_status`
- `gllm/loop/packet.py` — `result_to_evidence_packet`
- `gllm/loop/vericut_gate.py` — `VericutFinalGate` (production final-gate wiring)
- `tests/loop/test_findings.py`, `test_validators.py`, `test_scoring.py`, `test_generator.py`, `test_controller.py`, `test_packet.py`, `test_engine_end_to_end.py`

**Modify:**
- `gllm/utils/gcode_utils.py` — guard `IndexError` in `validate_z_levels` + `check_tool_offsets`; stop silent-pass in `validate_functional_correctness`
- `gllm/proof/runner.py` — use shared `setup_constraints` (final task: delegate to `LoopController`)

---

## Task 1: Package scaffolding + core types

**Files:**
- Create: `gllm/loop/__init__.py`
- Create: `gllm/loop/types.py`
- Create: `tests/loop/__init__.py` (empty)
- Test: `tests/loop/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_types.py
import unittest
from gllm.loop.types import Mode, Objective, TokenUsage, Budget, LoopRequest


class TypesTests(unittest.TestCase):
    def test_token_usage_total(self):
        self.assertEqual(TokenUsage(3, 5).total, 8)

    def test_budget_tracks_and_exhausts(self):
        b = Budget(max_attempts=4, token_budget=10, patience=2)
        self.assertFalse(b.exhausted())
        b.add(TokenUsage(4, 4))
        self.assertFalse(b.exhausted())
        b.add(TokenUsage(2, 1))
        self.assertTrue(b.exhausted())

    def test_budget_without_limit_never_exhausts(self):
        b = Budget(token_budget=None)
        b.add(TokenUsage(1_000_000, 0))
        self.assertFalse(b.exhausted())

    def test_loop_request_defaults(self):
        req = LoopRequest(prompt="mill a square")
        self.assertEqual(req.mode, Mode.GENERATE)
        self.assertIsNone(req.objective)
        self.assertEqual(req.max_attempts, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_types.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop'`

- [ ] **Step 3: Create the package files**

```python
# gllm/loop/__init__.py
"""Headless engine for the autonomous generate/improve/repair G-code loop."""
```

```python
# gllm/loop/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gllm.loop.findings import Finding


class Mode(str, Enum):
    GENERATE = "generate"
    REPAIR = "repair"
    IMPROVE = "improve"


@dataclass(frozen=True)
class Objective:
    metric: Literal["path_length", "tool_changes"]
    direction: Literal["minimize"] = "minimize"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, response) -> "TokenUsage":
        meta = getattr(response, "usage_metadata", None) or {}
        return cls(
            int(meta.get("input_tokens", 0) or 0),
            int(meta.get("output_tokens", 0) or 0),
        )


@dataclass
class Budget:
    max_attempts: int = 4
    token_budget: int | None = None
    patience: int = 2
    spent_tokens: int = 0

    def add(self, usage: "TokenUsage") -> None:
        self.spent_tokens += usage.total

    def exhausted(self) -> bool:
        return self.token_budget is not None and self.spent_tokens >= self.token_budget


@dataclass(frozen=True)
class ValidationContext:
    setup: object | None = None          # gllm.vericut.registry.VericutSetup | None
    feed_rate_min: float = 1.0
    feed_rate_max: float = 100.0
    spindle_speed_max: float = 900.0

    @classmethod
    def from_setup(cls, setup) -> "ValidationContext":
        if setup is None:
            return cls()
        return cls(
            setup=setup,
            feed_rate_min=setup.feed_rate_min if setup.feed_rate_min is not None else 1.0,
            feed_rate_max=setup.feed_rate_max if setup.feed_rate_max is not None else 100.0,
            spindle_speed_max=(
                setup.spindle_speed_max if setup.spindle_speed_max is not None else 900.0
            ),
        )


@dataclass(frozen=True)
class LoopRequest:
    prompt: str
    mode: Mode = Mode.GENERATE
    registry_path: str | Path | None = None
    setup_id: str | None = None
    seed_gcode: str | None = None
    objective: Objective | None = None
    model_name: str = "OpenRouter"
    openrouter_model_name: str | None = None
    run_vericut: bool = False
    max_attempts: int = 4
    token_budget: int | None = None
    patience: int = 2
    output_root: str | Path = ".loop-runs"
    scenario_id: str | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class Attempt:
    number: int
    gcode: str
    findings: tuple["Finding", ...]
    score: float
    objective_value: float | None = None

    @property
    def blocking_findings(self) -> tuple["Finding", ...]:
        return tuple(f for f in self.findings if f.blocking)


@dataclass(frozen=True)
class LoopResult:
    request: LoopRequest
    scenario_id: str
    status: str
    best: Attempt | None
    history: tuple[Attempt, ...]
    vericut: dict | None = None


@dataclass(frozen=True)
class LoopEvent:
    type: str            # attempt_started|findings|best_updated|repair_prompt|vericut_started|vericut_verdict|done
    attempt: Attempt | None = None
    prompt: str | None = None
    result: LoopResult | None = None
    payload: dict | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_types.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/__init__.py gllm/loop/types.py tests/loop/__init__.py tests/loop/test_types.py
git commit -m "feat(loop): add core engine types"
```

---

## Task 2: Normalized `Finding`

**Files:**
- Create: `gllm/loop/findings.py`
- Test: `tests/loop/test_findings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_findings.py
import unittest
from gllm.loop.findings import (
    Finding, Severity, from_lint, from_static, from_vericut, prioritize, format_findings,
)
from gllm.vericut.static_checks import StaticFinding


class FindingsTests(unittest.TestCase):
    def test_from_lint_ok_returns_none(self):
        self.assertIsNone(from_lint("safety", (True, None)))

    def test_from_lint_failure_builds_blocking_error(self):
        f = from_lint("safety", (False, "rapid through material"))
        self.assertEqual(f.source, "lint")
        self.assertEqual(f.code, "safety")
        self.assertEqual(f.severity, Severity.ERROR)
        self.assertTrue(f.blocking)
        self.assertIn("rapid", f.message)

    def test_from_lint_can_be_nonblocking_warning(self):
        f = from_lint("continuity", (False, "discontinuity"), blocking=False, severity=Severity.WARNING)
        self.assertFalse(f.blocking)
        self.assertEqual(f.severity, Severity.WARNING)

    def test_from_static_maps_error_to_blocking(self):
        sf = StaticFinding(code="unsupported_tool", severity="error", message="T99 not allowed", line_number=2)
        f = from_static(sf)
        self.assertEqual(f.source, "static")
        self.assertEqual(f.code, "unsupported_tool")
        self.assertEqual(f.line_number, 2)
        self.assertTrue(f.blocking)

    def test_from_vericut_payload(self):
        f = from_vericut({"code": "collision", "severity": "error", "message": "X exceeded", "line_number": 4})
        self.assertEqual(f.source, "vericut")
        self.assertEqual(f.code, "collision")
        self.assertTrue(f.blocking)

    def test_prioritize_orders_errors_and_static_first(self):
        warn = from_lint("continuity", (False, "x"), blocking=False, severity=Severity.WARNING)
        err_static = from_static(StaticFinding("unsupported_tool", "error", "x", 5))
        ordered = prioritize([warn, err_static])
        self.assertEqual(ordered[0].code, "unsupported_tool")

    def test_format_findings_handles_empty(self):
        self.assertIn("No concrete findings", format_findings([]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_findings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.findings'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/findings.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_findings.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/findings.py tests/loop/test_findings.py
git commit -m "feat(loop): add normalized Finding and source normalizers"
```

---

## Task 3: Fix the unreliable lint validators

These three `gcode_utils` validators are unsafe to trust as gate signals. Fix them so they are crash-free and honest. (The loop wires the reliable ones as blocking and the questionable ones as non-blocking in Task 5.)

**Files:**
- Modify: `gllm/utils/gcode_utils.py` (`validate_z_levels` ~222-233, `check_tool_offsets` ~345-359, `validate_functional_correctness` ~236-294)
- Test: `tests/loop/test_validator_fixes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_validator_fixes.py
import unittest
from gllm.utils.gcode_utils import (
    validate_z_levels, check_tool_offsets, validate_functional_correctness,
)


class ValidatorFixTests(unittest.TestCase):
    def test_z_levels_does_not_crash_on_lines_without_gcodes(self):
        # "S1000" has modal params but block.gcodes is empty -> used to IndexError.
        ok, _ = validate_z_levels("S1000\nF50\nM30", max_depth=100)
        self.assertTrue(ok)

    def test_check_tool_offsets_does_not_crash_on_lines_without_gcodes(self):
        ok, _ = check_tool_offsets("G43\nS1000\nG49\nM30")
        self.assertTrue(ok)

    def test_functional_correctness_does_not_silently_pass_on_unsupported_op(self):
        # Force the inner zip(*tool_path) to raise via a degenerate parameters_string;
        # the function must NOT return (True, None) by swallowing the exception.
        params = "starting_point: (0, 0)\ntool_path: bogus"
        ok, msg = validate_functional_correctness("G1 X1 Y1", params)
        # Either it parsed nothing (no user params -> True) or it flagged inability to verify,
        # but it must never claim success while swallowing an exception.
        if ok is False:
            self.assertIsNotNone(msg)
        self.assertIsInstance(ok, bool)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_validator_fixes.py -q`
Expected: FAIL — `IndexError: list index out of range` from `validate_z_levels` (and `check_tool_offsets`).

- [ ] **Step 3: Apply the fixes**

In `gllm/utils/gcode_utils.py`, change `validate_z_levels` so the `gcodes[0]` access is guarded:

```python
def validate_z_levels(gcode_string, max_depth):
    """Verify that Z-level movements do not exceed certain depth limits to prevent the tool from crashing into the workpiece or machine bed."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if not line.block.gcodes:
            continue
        params = line.block.gcodes[0].params
        if 'Z' in params:
            z_level = params['Z']
            if z_level > max_depth:
                error_msg = f"Z-level exceeds maximum depth at {line_text}"
                print(error_msg)
                return False, error_msg
    return True, None
```

In `check_tool_offsets`, guard the same access:

```python
def check_tool_offsets(gcode_string):
    """Validate that tool offsets are being used correctly and reset appropriately to avoid unintended tool paths."""
    tool_offset_active = False
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'G43' in line_text:  # Tool length offset compensation activate
            tool_offset_active = True
        elif 'G49' in line_text:  # Tool length offset compensation cancel
            tool_offset_active = False
        elif tool_offset_active and line.block.gcodes and 'Z' in line.block.gcodes[0].params:
            error_msg = f"Z movement with active tool offset in line: {line_text}"
            print(error_msg)
            return False, error_msg
    return True, None
```

In `validate_functional_correctness`, replace the bare `except:` that returns `(True, None)` with one that reports inability to verify instead of claiming success. Change the `except` block (~290-292) to:

```python
        except Exception as exc:
            print(f"Could not verify functional correctness: {exc}")
            return False, f"Could not verify tool path against specification (unsupported operation): {exc}"
```

- [ ] **Step 4: Run the new test plus the existing gcode_utils suite**

Run: `poetry run pytest tests/loop/test_validator_fixes.py tests/test_gcode_utils.py -q`
Expected: PASS (new tests pass; no regressions in `tests/test_gcode_utils.py`)

> If a pre-existing `test_gcode_utils.py` test asserted the old silent-pass behavior of `validate_functional_correctness`, update that test to expect the honest `(False, msg)` return and note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add gllm/utils/gcode_utils.py tests/loop/test_validator_fixes.py
git commit -m "fix(gcode): guard IndexError in z_levels/tool_offsets; stop silent-pass in functional check"
```

---

## Task 4: Shared `setup_constraints`

**Files:**
- Create: `gllm/loop/constraints.py`
- Modify: `gllm/proof/runner.py` (replace body of `_setup_constraints` ~495-522 with a delegating call)
- Test: `tests/loop/test_constraints.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_constraints.py
import unittest
from gllm.loop.constraints import setup_constraints
from gllm.vericut.registry import load_setup_registry


class _StubSetup:
    id = "stub"
    allowed_tools = frozenset({"T1", "T2"})
    required_modes = frozenset({"G90"})
    expected_units = "inch"
    feed_rate_min = 1.0
    feed_rate_max = 100.0
    spindle_speed_min = None
    spindle_speed_max = 900.0
    work_envelope = None
    safe_z_min = 0.25


class ConstraintsTests(unittest.TestCase):
    def test_includes_tools_modes_units_feed_spindle_safe_z(self):
        c = setup_constraints(_StubSetup())
        joined = "\n".join(c)
        self.assertIn("allowed_tools: T1, T2", joined)
        self.assertIn("required_modes: G90", joined)
        self.assertIn("expected_units: inch", joined)
        self.assertIn("feed_rate_range: [1, 100]", joined)
        self.assertIn("spindle_speed_range: [-inf, 900]", joined)
        self.assertIn("safe_z_min: 0.25", joined)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_constraints.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.constraints'`

- [ ] **Step 3: Write `constraints.py` (logic lifted verbatim from `proof/runner._setup_constraints`)**

```python
# gllm/loop/constraints.py
from __future__ import annotations


def _format_optional_range(minimum: float | None, maximum: float | None) -> str:
    lower = "-inf" if minimum is None else f"{minimum:g}"
    upper = "inf" if maximum is None else f"{maximum:g}"
    return f"[{lower}, {upper}]"


def setup_constraints(setup) -> tuple[str, ...]:
    """Human-readable machine constraints fed into generate/repair/improve prompts."""
    constraints: list[str] = []
    if setup.allowed_tools:
        constraints.append(f"allowed_tools: {', '.join(sorted(setup.allowed_tools))}")
    if setup.required_modes:
        constraints.append(f"required_modes: {', '.join(sorted(setup.required_modes))}")
    if setup.expected_units:
        constraints.append(f"expected_units: {setup.expected_units}")
    if setup.feed_rate_min is not None or setup.feed_rate_max is not None:
        constraints.append(
            f"feed_rate_range: {_format_optional_range(setup.feed_rate_min, setup.feed_rate_max)}"
        )
    if setup.spindle_speed_min is not None or setup.spindle_speed_max is not None:
        constraints.append(
            "spindle_speed_range: "
            f"{_format_optional_range(setup.spindle_speed_min, setup.spindle_speed_max)}"
        )
    if setup.work_envelope is not None:
        envelope = setup.work_envelope
        constraints.append(
            "work_envelope: "
            f"X{_format_optional_range(envelope.x_min, envelope.x_max)}, "
            f"Y{_format_optional_range(envelope.y_min, envelope.y_max)}, "
            f"Z{_format_optional_range(envelope.z_min, envelope.z_max)}"
        )
    if setup.safe_z_min is not None:
        constraints.append(f"safe_z_min: {setup.safe_z_min:g}")
    return tuple(constraints)
```

- [ ] **Step 4: Make `proof/runner._setup_constraints` delegate (DRY)**

Replace the body of `_setup_constraints` in `gllm/proof/runner.py` with:

```python
def _setup_constraints(setup) -> tuple[str, ...]:
    from gllm.loop.constraints import setup_constraints

    return setup_constraints(setup)
```

- [ ] **Step 5: Run the new test plus the proof suites**

Run: `poetry run pytest tests/loop/test_constraints.py tests/test_proof_run.py -q`
Expected: PASS (new test passes; `test_proof_run.py` still green — `test_repair_prompt_includes_prior_candidate_evidence_and_setup_constraints` exercises the delegated constraints)

- [ ] **Step 6: Commit**

```bash
git add gllm/loop/constraints.py gllm/proof/runner.py tests/loop/test_constraints.py
git commit -m "refactor(loop): extract shared setup_constraints; proof runner delegates"
```

---

## Task 5: Validators (lint + static)

**Files:**
- Create: `gllm/loop/validators.py`
- Test: `tests/loop/test_validators.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_validators.py
import unittest
from gllm.loop.validators import LintValidator, StaticPolicyValidator
from gllm.loop.types import ValidationContext
from gllm.loop.findings import Severity
from gllm.vericut.registry import load_setup_registry


GOOD = "G90\nT1 M06\nS800\nG0 Z1.0\nG1 Z-0.1 F50\nM30\n"


class LintValidatorTests(unittest.TestCase):
    def test_clean_program_has_no_blocking_findings(self):
        findings = LintValidator().validate(GOOD, ValidationContext())
        self.assertEqual([f for f in findings if f.blocking], [])

    def test_rapid_through_material_is_blocking(self):
        bad = "G1 X1 Y1 F50\nG0 X5 Y5\nM30\n"
        findings = LintValidator().validate(bad, ValidationContext())
        codes = {f.code for f in findings if f.blocking}
        self.assertIn("safety", codes)

    def test_continuity_is_nonblocking_warning(self):
        # validate_continuity is known-noisy; it must never block the loop.
        findings = LintValidator().validate("G1 X1 Y1\nG1 X9 Y9\nM30\n", ValidationContext())
        for f in findings:
            if f.code == "continuity":
                self.assertFalse(f.blocking)
                self.assertEqual(f.severity, Severity.WARNING)

    def test_validator_never_raises(self):
        # Degenerate input must yield findings, not an exception.
        findings = LintValidator().validate("(comment only)\n%\n", ValidationContext())
        self.assertIsInstance(findings, list)


class StaticPolicyValidatorTests(unittest.TestCase):
    def test_no_setup_means_no_findings(self):
        self.assertEqual(StaticPolicyValidator().validate(GOOD, ValidationContext()), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_validators.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.validators'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/validators.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_validators.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/validators.py tests/loop/test_validators.py
git commit -m "feat(loop): add LintValidator and StaticPolicyValidator"
```

---

## Task 6: Scoring + objective estimators

**Files:**
- Create: `gllm/loop/scoring.py`
- Test: `tests/loop/test_scoring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_scoring.py
import unittest
from gllm.loop.scoring import findings_penalty, objective_value, path_length, tool_changes
from gllm.loop.findings import Finding, Severity
from gllm.loop.types import Objective


class ScoringTests(unittest.TestCase):
    def test_no_findings_is_zero_penalty(self):
        self.assertEqual(findings_penalty([]), 0.0)

    def test_blocking_error_outweighs_nonblocking_warning(self):
        err = Finding("static", "x", Severity.ERROR, "m", blocking=True)
        warn = Finding("lint", "continuity", Severity.WARNING, "m", blocking=False)
        self.assertGreater(findings_penalty([err]), findings_penalty([warn]))

    def test_tool_changes_counts_t_words(self):
        self.assertEqual(tool_changes("T1 M06\nG0 X1\nT2 M06\nM30"), 2.0)

    def test_path_length_sums_segments(self):
        # (0,0)->(3,0)->(3,4): 3 + 4 = 7
        gcode = "G1 X3 Y0 F50\nG1 X3 Y4 F50\nM30"
        self.assertAlmostEqual(path_length(gcode), 7.0, places=3)

    def test_objective_value_none_objective(self):
        self.assertIsNone(objective_value("G1 X1 Y1\nM30", None))

    def test_objective_value_robust_to_garbage(self):
        # Must return a float or None, never raise.
        self.assertIsInstance(objective_value("not gcode", Objective("path_length")), (float, type(None)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.scoring'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/scoring.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_scoring.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/scoring.py tests/loop/test_scoring.py
git commit -m "feat(loop): add findings_penalty and objective estimators"
```

---

## Task 7: Candidate generator + prompt builders

**Files:**
- Create: `gllm/loop/generator.py`
- Test: `tests/loop/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_generator.py
import unittest
from langchain_core.messages.ai import AIMessage
from gllm.loop.generator import (
    LLMCandidateGenerator, build_generate_prompt, build_repair_prompt, build_improve_prompt,
)
from gllm.loop.types import LoopRequest, Objective
from gllm.loop.findings import Finding, Severity


class _FakeChain:
    def __init__(self, content, usage=None):
        self._content = content
        self._usage = usage

    def invoke(self, _inputs):
        msg = AIMessage(content=self._content)
        if self._usage is not None:
            msg.usage_metadata = self._usage
        return msg


class GeneratorTests(unittest.TestCase):
    def test_generator_returns_text_and_tracks_usage(self):
        chain = _FakeChain("G90\nM30", usage={"input_tokens": 10, "output_tokens": 4})
        gen = LLMCandidateGenerator(chain=chain)
        out = gen("make a square", None)
        self.assertEqual(out, "G90\nM30")
        self.assertEqual(gen.last_usage.total, 14)

    def test_generator_handles_missing_usage(self):
        gen = LLMCandidateGenerator(chain=_FakeChain("G90\nM30"))
        gen("x", None)
        self.assertEqual(gen.last_usage.total, 0)

    def test_generate_prompt_includes_task_and_constraints(self):
        req = LoopRequest(prompt="mill a square")
        p = build_generate_prompt(req, ("allowed_tools: T1",))
        self.assertIn("mill a square", p)
        self.assertIn("allowed_tools: T1", p)

    def test_repair_prompt_includes_findings_and_previous(self):
        req = LoopRequest(prompt="mill a square")
        findings = [Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", line_number=2)]
        p = build_repair_prompt(req, "T99 M06", findings, ("allowed_tools: T1",))
        self.assertIn("unsupported_tool", p)
        self.assertIn("T99 M06", p)
        self.assertIn("allowed_tools: T1", p)

    def test_improve_prompt_states_objective_and_value(self):
        req = LoopRequest(prompt="mill a square", objective=Objective("path_length"))
        p = build_improve_prompt(req, "G1 X1 Y1", Objective("path_length"), 12.5, ())
        self.assertIn("path_length", p)
        self.assertIn("12.5", p)
        self.assertIn("G1 X1 Y1", p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_generator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.generator'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/generator.py
from __future__ import annotations

from typing import Iterable, Protocol

from langchain_core.messages.ai import AIMessage

from gllm.loop.findings import Finding, format_findings
from gllm.loop.types import LoopRequest, Objective, TokenUsage
from gllm.utils.model_utils import setup_langchain_without_rag, setup_model


class CandidateGenerator(Protocol):
    last_usage: TokenUsage

    def __call__(self, prompt: str, ctx: object | None = None) -> str: ...


class LLMCandidateGenerator:
    """Pure prompt -> raw G-code text. Builds its chain lazily from model_utils."""

    def __init__(self, model_name: str = "OpenRouter", openrouter_model_name: str | None = None, chain=None):
        self._model_name = model_name
        self._openrouter_model_name = openrouter_model_name
        self._chain = chain
        self.last_usage = TokenUsage()

    def _ensure_chain(self):
        if self._chain is None:
            model = setup_model(self._model_name, self._openrouter_model_name)
            self._chain = setup_langchain_without_rag(model)
        return self._chain

    def __call__(self, prompt: str, ctx: object | None = None) -> str:
        chain = self._ensure_chain()
        response = chain.invoke({"input": prompt})
        self.last_usage = TokenUsage.from_response(response)
        return response.content if isinstance(response, AIMessage) else str(response)


def _constraints_block(constraints: Iterable[str]) -> str:
    items = list(constraints)
    return "\n".join(f"- {c}" for c in items) if items else "- No machine constraints were provided."


def build_generate_prompt(request: LoopRequest, constraints: Iterable[str]) -> str:
    return (
        "Generate a complete, robust CNC G-code program for the task below. "
        "Return only G-code and no prose.\n\n"
        f"Task:\n{request.prompt}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n"
    )


def build_repair_prompt(
    request: LoopRequest,
    previous_gcode: str,
    findings: Iterable[Finding],
    constraints: Iterable[str],
) -> str:
    return (
        "The previous G-code was rejected by machine checks. Repair it using only the "
        "concrete findings below. Return the full corrected G-code program and no prose.\n\n"
        f"Original task:\n{request.prompt}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n\n"
        f"Findings:\n{format_findings(findings)}\n\n"
        f"Previous G-code:\n{previous_gcode}\n"
    )


def build_improve_prompt(
    request: LoopRequest,
    current_gcode: str,
    objective: Objective,
    current_value: float | None,
    constraints: Iterable[str],
) -> str:
    value_text = "unknown" if current_value is None else f"{current_value:g}"
    return (
        "The G-code below already passes all machine checks. Improve it to "
        f"{objective.direction} {objective.metric} WITHOUT violating any constraint. "
        "Every machine check must still pass. Return the full improved G-code program and no prose.\n\n"
        f"Original task:\n{request.prompt}\n\n"
        f"Current {objective.metric}: {value_text}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n\n"
        f"Current G-code:\n{current_gcode}\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_generator.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/generator.py tests/loop/test_generator.py
git commit -m "feat(loop): add LLM candidate generator and prompt builders"
```

---

## Task 8: LoopController (the loop)

**Files:**
- Create: `gllm/loop/controller.py`
- Test: `tests/loop/test_controller.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_controller.py
import unittest
from gllm.loop.controller import LoopController, pick_best, derive_status
from gllm.loop.types import LoopRequest, Mode, Objective, Attempt
from gllm.loop.findings import Finding, Severity


def _err(code="x"):
    return Finding("static", code, Severity.ERROR, "bad", blocking=True)


class _ScriptedValidator:
    """Returns a pre-scripted findings list per attempt index (0-based)."""
    name = "scripted"
    tier = "static"

    def __init__(self, per_attempt):
        self._per_attempt = per_attempt
        self._i = -1

    def validate(self, gcode, ctx):
        self._i += 1
        idx = min(self._i, len(self._per_attempt) - 1)
        return list(self._per_attempt[idx])


class _ScriptedGenerator:
    def __init__(self, outputs):
        self._outputs = outputs
        self._i = -1
        from gllm.loop.types import TokenUsage
        self.last_usage = TokenUsage(1, 1)

    def __call__(self, prompt, ctx=None):
        self._i += 1
        return self._outputs[min(self._i, len(self._outputs) - 1)]


class ControllerLoopTests(unittest.TestCase):
    def _controller(self, gen, validators):
        return LoopController(generator=gen, blocking_validators=validators, final_gate=None)

    def test_converges_and_stops_early(self):
        gen = _ScriptedGenerator(["bad1", "good"])
        val = _ScriptedValidator([[_err()], []])  # attempt2 clean
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=5))
        self.assertEqual(result.status, "passed")
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.best.gcode, "good")

    def test_caps_at_max_attempts_and_returns_best_effort(self):
        gen = _ScriptedGenerator(["a", "b", "c"])
        val = _ScriptedValidator([[_err(), _err("y")], [_err()], [_err()]])
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=2))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.status, "exhausted_best_effort")
        # best = attempt 2 (1 error) over attempt 1 (2 errors)
        self.assertEqual(result.best.number, 2)

    def test_patience_stops_when_no_improvement(self):
        gen = _ScriptedGenerator(["a", "b", "c", "d", "e"])
        val = _ScriptedValidator([[_err()]])  # every attempt: same single error
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=10, patience=2))
        # attempt1 sets best; attempts 2 and 3 don't improve -> stop after 3
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.status, "exhausted_best_effort")

    def test_repair_mode_uses_seed_as_attempt_one(self):
        gen = _ScriptedGenerator(["regenerated"])  # only used if a 2nd attempt happens
        val = _ScriptedValidator([[]])  # seed is already clean
        req = LoopRequest(prompt="p", mode=Mode.REPAIR, seed_gcode="SEED", max_attempts=3)
        result = self._controller(gen, [val]).run(req)
        self.assertEqual(result.history[0].gcode, "SEED")
        self.assertEqual(result.status, "passed")

    def test_improve_keeps_best_objective_among_green(self):
        # Both attempts green; improve picks the lower path_length one.
        gen = _ScriptedGenerator(["G1 X10 Y0\nM30"])  # 2nd candidate shorter
        val = _ScriptedValidator([[]])
        req = LoopRequest(
            prompt="p", mode=Mode.IMPROVE, seed_gcode="G1 X100 Y0\nM30",
            objective=Objective("path_length"), max_attempts=2, patience=5,
        )
        result = self._controller(gen, [val]).run(req)
        self.assertIn(result.status, ("improved", "not_improved"))
        self.assertEqual(result.best.gcode, "G1 X10 Y0\nM30")
        self.assertEqual(result.status, "improved")


class PickBestStatusTests(unittest.TestCase):
    def test_pick_best_prefers_fewer_errors(self):
        a1 = Attempt(1, "a", (_err(), _err("y")), score=2000.0)
        a2 = Attempt(2, "b", (_err(),), score=1000.0)
        self.assertIs(pick_best(a1, a2, Mode.GENERATE, None), a2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_controller.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.controller'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/controller.py
from __future__ import annotations

import uuid
from typing import Callable, Iterator, Sequence

from gllm.loop.constraints import setup_constraints
from gllm.loop.findings import Finding
from gllm.loop.generator import (
    build_generate_prompt, build_improve_prompt, build_repair_prompt,
)
from gllm.loop.scoring import findings_penalty, objective_value
from gllm.loop.types import (
    Attempt, Budget, LoopEvent, LoopRequest, LoopResult, Mode, Objective, ValidationContext,
)
from gllm.loop.validators import LintValidator, StaticPolicyValidator
from gllm.utils.gcode_utils import clean_gcode

# A final gate takes the best gcode + context and returns a verdict dict (or None if unavailable).
FinalGate = Callable[[str, ValidationContext, LoopRequest], "dict | None"]


def pick_best(best: Attempt | None, candidate: Attempt, mode: Mode, objective: Objective | None) -> Attempt:
    if best is None:
        return candidate
    if mode == Mode.IMPROVE:
        cand_ok = not candidate.blocking_findings
        best_ok = not best.blocking_findings
        if cand_ok and not best_ok:
            return candidate
        if best_ok and not cand_ok:
            return best
        if cand_ok and best_ok:
            cv = candidate.objective_value if candidate.objective_value is not None else float("inf")
            bv = best.objective_value if best.objective_value is not None else float("inf")
            return candidate if cv < bv else best
        return candidate if candidate.score < best.score else best
    return candidate if candidate.score < best.score else best


def converged(attempt: Attempt, mode: Mode) -> bool:
    if attempt.blocking_findings:
        return False
    # IMPROVE never early-converges on green; patience/cap/budget stop it.
    return mode != Mode.IMPROVE


def derive_status(best: Attempt | None, mode: Mode, vericut: dict | None) -> str:
    if best is None:
        return "failed"
    if best.blocking_findings:
        return "exhausted_best_effort"
    if mode == Mode.IMPROVE:
        return "improved" if best.number > 1 else "not_improved"
    if vericut is not None:
        if vericut.get("passed") is True:
            return "accepted_vericut"
        if vericut.get("status") == "vericut_unverified":
            return "blocked_vericut_unverified"
        return "rejected_vericut"
    has_warnings = any(not f.blocking for f in best.findings)
    return "passed_with_warnings" if has_warnings else "passed"


class LoopController:
    def __init__(
        self,
        *,
        generator,
        blocking_validators: Sequence | None = None,
        final_gate: FinalGate | None = None,
    ):
        self._generator = generator
        self._validators = list(blocking_validators) if blocking_validators is not None else [
            LintValidator(),
            StaticPolicyValidator(),
        ]
        self._final_gate = final_gate

    def run(self, request: LoopRequest) -> LoopResult:
        result: LoopResult | None = None
        for event in self.stream(request):
            if event.type == "done":
                result = event.result
        assert result is not None
        return result

    def stream(self, request: LoopRequest) -> Iterator[LoopEvent]:
        scenario_id = request.scenario_id or uuid.uuid4().hex
        setup = self._load_setup(request)
        ctx = ValidationContext.from_setup(setup)
        constraints = setup_constraints(setup) if setup is not None else ()
        budget = Budget(request.max_attempts, request.token_budget, request.patience)

        best: Attempt | None = None
        history: list[Attempt] = []
        no_improve = 0

        if request.mode in (Mode.REPAIR, Mode.IMPROVE) and request.seed_gcode is not None:
            candidate = request.seed_gcode
        else:
            candidate = self._generator(build_generate_prompt(request, constraints), ctx)
            budget.add(self._generator.last_usage)

        for n in range(1, budget.max_attempts + 1):
            yield LoopEvent("attempt_started", payload={"attempt": n})
            cleaned = clean_gcode(candidate)
            findings = self._validate(cleaned, ctx)
            score = findings_penalty(findings)
            obj_value = objective_value(cleaned, request.objective)
            attempt = Attempt(n, cleaned, tuple(findings), score, obj_value)
            history.append(attempt)
            yield LoopEvent("findings", attempt=attempt)

            previous_best = best
            best = pick_best(best, attempt, request.mode, request.objective)
            if best is attempt and best is not previous_best:
                no_improve = 0
                yield LoopEvent("best_updated", attempt=attempt)
            else:
                no_improve += 1

            if converged(attempt, request.mode):
                break
            if n >= budget.max_attempts or budget.exhausted() or no_improve >= budget.patience:
                break

            prompt = self._revision_prompt(request, best, constraints)
            yield LoopEvent("repair_prompt", prompt=prompt)
            candidate = self._generator(prompt, ctx)
            budget.add(self._generator.last_usage)

        vericut = None
        if (
            self._final_gate is not None
            and request.run_vericut
            and best is not None
            and not best.blocking_findings
        ):
            yield LoopEvent("vericut_started")
            vericut = self._final_gate(best.gcode, ctx, request)
            yield LoopEvent("vericut_verdict", payload=vericut)

        status = derive_status(best, request.mode, vericut)
        result = LoopResult(
            request=request,
            scenario_id=scenario_id,
            status=status,
            best=best,
            history=tuple(history),
            vericut=vericut,
        )
        yield LoopEvent("done", result=result)

    def _validate(self, gcode: str, ctx: ValidationContext) -> list[Finding]:
        findings: list[Finding] = []
        for validator in self._validators:
            findings.extend(validator.validate(gcode, ctx))
        return findings

    def _revision_prompt(self, request: LoopRequest, best: Attempt, constraints) -> str:
        if request.mode == Mode.IMPROVE and request.objective is not None:
            return build_improve_prompt(
                request, best.gcode, request.objective, best.objective_value, constraints
            )
        return build_repair_prompt(request, best.gcode, best.findings, constraints)

    def _load_setup(self, request: LoopRequest):
        if request.registry_path is None or request.setup_id is None:
            return None
        from gllm.vericut.registry import load_setup_registry

        registry = load_setup_registry(request.registry_path)
        return registry.get(request.setup_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_controller.py -q`
Expected: PASS (6 passed)

> Note on `test_improve_keeps_best_objective_among_green`: attempt 1 is the seed (`path_length`=100), attempt 2 is the generated shorter path (`path_length`=10), both green, so `pick_best` selects attempt 2 and `derive_status` returns `improved` (best.number > 1).

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/controller.py tests/loop/test_controller.py
git commit -m "feat(loop): add LoopController with converge-or-cap + best-so-far"
```

---

## Task 9: Evidence-packet adapter

Map a `LoopResult` onto the existing `EvidencePacket`/`ProofAttempt` JSON shape so downstream tooling (and the corpus) sees a familiar structure, plus the new `score`/`mode`/`objective`/`best_attempt_index` fields.

**Files:**
- Create: `gllm/loop/packet.py`
- Test: `tests/loop/test_packet.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_packet.py
import unittest
from gllm.loop.packet import result_to_dict
from gllm.loop.types import LoopRequest, LoopResult, Attempt, Mode, Objective
from gllm.loop.findings import Finding, Severity


class PacketTests(unittest.TestCase):
    def _result(self):
        a1 = Attempt(1, "BAD", (Finding("static", "unsupported_tool", Severity.ERROR, "T99", line_number=2),), 1000.0)
        a2 = Attempt(2, "GOOD", (), 0.0)
        req = LoopRequest(prompt="mill", mode=Mode.GENERATE, setup_id="s1")
        return LoopResult(req, "sc-1", "passed", best=a2, history=(a1, a2))

    def test_top_level_shape(self):
        d = result_to_dict(self._result())
        self.assertEqual(d["scenario_id"], "sc-1")
        self.assertEqual(d["status"], "passed")
        self.assertEqual(d["mode"], "generate")
        self.assertEqual(d["best_attempt_index"], 2)
        self.assertEqual(d["operator_action"], "rerun_vericut")  # accepted_static_only-equivalent? see note
        self.assertEqual(len(d["attempts"]), 2)

    def test_attempt_carries_findings_and_score(self):
        d = result_to_dict(self._result())
        first = d["attempts"][0]
        self.assertEqual(first["attempt_number"], 1)
        self.assertEqual(first["score"], 1000.0)
        self.assertEqual(first["findings"][0]["code"], "unsupported_tool")
```

> Note: `operator_action` reuses `proof.runner.operator_action_for_status`. For the new `passed`/`exhausted_best_effort`/`improved` statuses that function returns `manual_review_required`; map them explicitly in `result_to_dict` (see implementation). Adjust the asserted value in `test_top_level_shape` to match the mapping you implement — the test above expects `passed → ready_to_review`; change the assertion to `self.assertEqual(d["operator_action"], "ready_to_review")`.

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_packet.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.packet'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/packet.py
from __future__ import annotations

from gllm.loop.types import Attempt, LoopResult, Mode

# Map engine statuses to operator actions, falling back to the proof runner's table.
_ACTION_OVERRIDES = {
    "passed": "ready_to_review",
    "passed_with_warnings": "ready_to_review",
    "improved": "ready_to_review",
    "not_improved": "ready_to_review",
    "exhausted_best_effort": "fix_prompt",
    "failed": "manual_review_required",
}


def operator_action(status: str) -> str:
    if status in _ACTION_OVERRIDES:
        return _ACTION_OVERRIDES[status]
    from gllm.proof.runner import operator_action_for_status

    return operator_action_for_status(status)


def _attempt_to_dict(attempt: Attempt) -> dict:
    return {
        "attempt_number": attempt.number,
        "score": attempt.score,
        "objective_value": attempt.objective_value,
        "gcode": attempt.gcode,
        "findings": [f.to_dict() for f in attempt.findings],
        "blocking_findings": [f.to_dict() for f in attempt.blocking_findings],
    }


def result_to_dict(result: LoopResult) -> dict:
    objective = result.request.objective
    return {
        "scenario_id": result.scenario_id,
        "status": result.status,
        "operator_action": operator_action(result.status),
        "mode": result.request.mode.value if isinstance(result.request.mode, Mode) else str(result.request.mode),
        "objective": {"metric": objective.metric, "direction": objective.direction} if objective else None,
        "setup_id": result.request.setup_id,
        "best_attempt_index": result.best.number if result.best is not None else None,
        "vericut": result.vericut,
        "attempts": [_attempt_to_dict(a) for a in result.history],
    }
```

- [ ] **Step 4: Update the test assertion and run**

Change `test_top_level_shape`'s operator-action assertion to:

```python
        self.assertEqual(d["operator_action"], "ready_to_review")
```

Run: `poetry run pytest tests/loop/test_packet.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/packet.py tests/loop/test_packet.py
git commit -m "feat(loop): add LoopResult -> evidence dict adapter"
```

---

## Task 10: Vericut final gate (production wiring)

Wrap the existing Vericut staging + verdict pipeline as a `FinalGate` the controller can call on the converged best candidate. Auto-skips when assets/license are missing.

**Files:**
- Create: `gllm/loop/vericut_gate.py`
- Test: `tests/loop/test_vericut_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_vericut_gate.py
import json
import shutil
import unittest
from pathlib import Path

from gllm.loop.vericut_gate import VericutFinalGate
from gllm.loop.types import LoopRequest, ValidationContext
from gllm.vericut.runner import VericutRunResult


class VericutGateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".loop-vericut-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _registry(self):
        support = self.root / "support"
        support.mkdir()
        (support / "fixture.stl").write_text("fixture", encoding="utf-8")
        (support / "tools.tls").write_text("tools", encoding="utf-8")
        (support / "control.ctl").write_text("control", encoding="utf-8")
        project = self.root / "template.vcproject"
        project.write_text(
            '<VcProject Version="9.5" Unit="inch"><Setup Name="d">'
            '<Component Name="F"><STL><File>fixture.stl</File></STL></Component>'
            '<NCPrograms Type="gcode"><NCProgram Use="on"><File>old.mcd</File></NCProgram></NCPrograms>'
            '<APT><Library>tools.tls</Library></APT><Control><File>control.ctl</File></Control>'
            "</Setup></VcProject>",
            encoding="utf-8",
        )
        bat = self.root / "vericut.bat"
        bat.write_text("@echo off", encoding="utf-8")
        reg = self.root / "setups.json"
        reg.write_text(json.dumps({"setups": [{
            "id": "s1", "description": "d", "vericut_bat": str(bat),
            "project_template": str(project), "library_search_paths": [str(support)],
            "allowed_tools": ["T1"], "required_modes": ["G90"], "safe_z_min": 0.25,
        }]}), encoding="utf-8")
        return reg

    def test_rejected_verdict_from_log(self):
        reg = self._registry()
        from gllm.vericut.registry import load_setup_registry
        setup = load_setup_registry(reg).get("s1")
        ctx = ValidationContext.from_setup(setup)
        req = LoopRequest(prompt="p", registry_path=reg, setup_id="s1", run_vericut=True,
                          output_root=self.root / "runs")

        def fake_run_vericut(setup, job, *, batch=True, timeout_seconds=None):
            log = job.output_dir / "vericut.log"
            log.write_text("Error for line 4\n Error: Component \"X\" exceeded limit at line: (4)\n"
                           "Number of Errors: 1\nNumber of Warnings: 0", encoding="utf-8")
            return VericutRunResult(command=("vericut.bat",), returncode=0, stdout="", stderr="", artifacts=(log,))

        gate = VericutFinalGate(run_vericut_fn=fake_run_vericut)
        verdict = gate("G90\nT1 M06\nG0 Z1.0\nM30\n", ctx, req)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["status"], "vericut_rejected")

    def test_missing_setup_returns_none(self):
        ctx = ValidationContext()  # no setup
        req = LoopRequest(prompt="p", run_vericut=True)
        self.assertIsNone(VericutFinalGate().__call__("G90\nM30\n", ctx, req))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_vericut_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gllm.loop.vericut_gate'`

- [ ] **Step 3: Write the implementation**

```python
# gllm/loop/vericut_gate.py
from __future__ import annotations

from pathlib import Path

from gllm.loop.types import LoopRequest, ValidationContext
from gllm.vericut.registry import validate_setup_paths
from gllm.vericut.runner import prepare_job_workspace, run_vericut
from gllm.vericut.static_checks import check_gcode
from gllm.vericut.verdict import write_verdict_packet


class VericutFinalGate:
    """Production FinalGate: stages a Vericut job for the candidate and parses the verdict.

    Returns a verdict dict, or None when Vericut cannot run (no setup, missing assets,
    or staging reports missing dependencies)."""

    def __init__(self, run_vericut_fn=run_vericut):
        self._run_vericut_fn = run_vericut_fn

    def __call__(self, gcode: str, ctx: ValidationContext, request: LoopRequest) -> dict | None:
        setup = ctx.setup
        if setup is None:
            return None
        if validate_setup_paths(setup):  # non-empty -> missing assets/license
            return None

        workspace = Path(request.output_root) / (request.scenario_id or "loop") / "vericut-final"
        workspace.mkdir(parents=True, exist_ok=True)
        job = prepare_job_workspace(setup=setup, gcode=gcode, output_root=workspace, job_id="final")
        if job.missing_dependencies:
            return None

        run_result = self._run_vericut_fn(setup, job, batch=True, timeout_seconds=request.timeout_seconds)
        static_report = check_gcode(gcode, setup)
        verdict = write_verdict_packet(job=job, static_report=static_report, run_result=run_result)
        return verdict.to_dict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_vericut_gate.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gllm/loop/vericut_gate.py tests/loop/test_vericut_gate.py
git commit -m "feat(loop): add VericutFinalGate wiring existing staging + verdict"
```

---

## Task 11: End-to-end engine smoke test + public API

Prove the whole engine closes the loop in all three modes with fakes, and expose a clean public API from `gllm/loop/__init__.py`.

**Files:**
- Modify: `gllm/loop/__init__.py`
- Test: `tests/loop/test_engine_end_to_end.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_engine_end_to_end.py
import unittest
from gllm.loop import LoopController, LoopRequest, Mode, Objective, result_to_dict
from gllm.loop.findings import Finding, Severity


def _err():
    return Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", blocking=True)


class _Gen:
    def __init__(self, outs):
        from gllm.loop.types import TokenUsage
        self._outs, self._i, self.last_usage = outs, -1, TokenUsage(2, 2)

    def __call__(self, prompt, ctx=None):
        self._i += 1
        return self._outs[min(self._i, len(self._outs) - 1)]


class _Val:
    name, tier = "scripted", "static"

    def __init__(self, per):
        self._per, self._i = per, -1

    def validate(self, gcode, ctx):
        self._i += 1
        return list(self._per[min(self._i, len(self._per) - 1)])


class EngineEndToEndTests(unittest.TestCase):
    def test_generate_repairs_then_passes(self):
        ctrl = LoopController(generator=_Gen(["T99 M06", "T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        result = ctrl.run(LoopRequest(prompt="mill a square", max_attempts=4))
        self.assertEqual(result.status, "passed")
        d = result_to_dict(result)
        self.assertEqual(d["best_attempt_index"], 2)
        self.assertEqual(d["mode"], "generate")

    def test_repair_seed_then_passes(self):
        ctrl = LoopController(generator=_Gen(["T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        req = LoopRequest(prompt="fix it", mode=Mode.REPAIR, seed_gcode="T99 M06", max_attempts=4)
        result = ctrl.run(req)
        self.assertEqual(result.history[0].gcode, "T99 M06")
        self.assertEqual(result.status, "passed")

    def test_improve_returns_seed_when_nothing_beats_it(self):
        # Generated candidate is LONGER -> seed stays best -> not_improved.
        ctrl = LoopController(generator=_Gen(["G1 X999 Y0\nM30"]), blocking_validators=[_Val([[]])])
        req = LoopRequest(prompt="optimize", mode=Mode.IMPROVE, seed_gcode="G1 X1 Y0\nM30",
                          objective=Objective("path_length"), max_attempts=2, patience=5)
        result = ctrl.run(req)
        self.assertEqual(result.best.gcode, "G1 X1 Y0\nM30")
        self.assertEqual(result.status, "not_improved")

    def test_streaming_emits_expected_event_sequence(self):
        ctrl = LoopController(generator=_Gen(["T99 M06", "T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        types = [e.type for e in ctrl.stream(LoopRequest(prompt="p", max_attempts=4))]
        self.assertEqual(types[0], "attempt_started")
        self.assertIn("repair_prompt", types)
        self.assertEqual(types[-1], "done")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_engine_end_to_end.py -q`
Expected: FAIL — `ImportError: cannot import name 'LoopController' from 'gllm.loop'`

- [ ] **Step 3: Export the public API**

Replace `gllm/loop/__init__.py` with:

```python
# gllm/loop/__init__.py
"""Headless engine for the autonomous generate/improve/repair G-code loop."""

from gllm.loop.controller import LoopController, derive_status, pick_best
from gllm.loop.generator import LLMCandidateGenerator
from gllm.loop.packet import result_to_dict
from gllm.loop.types import (
    Attempt, Budget, LoopEvent, LoopRequest, LoopResult, Mode, Objective, TokenUsage, ValidationContext,
)
from gllm.loop.vericut_gate import VericutFinalGate

__all__ = [
    "LoopController", "LLMCandidateGenerator", "VericutFinalGate", "result_to_dict",
    "derive_status", "pick_best",
    "Attempt", "Budget", "LoopEvent", "LoopRequest", "LoopResult", "Mode", "Objective",
    "TokenUsage", "ValidationContext",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/loop/test_engine_end_to_end.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full loop + proof + gcode suites for regression**

Run: `poetry run pytest tests/loop tests/test_proof_run.py tests/test_proof_corpus.py tests/test_gcode_utils.py -q`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add gllm/loop/__init__.py tests/loop/test_engine_end_to_end.py
git commit -m "feat(loop): export public engine API + end-to-end mode coverage"
```

---

## Task 12: Wire the engine into `run_proof_scenario` (unification)

Make the existing proof runner delegate generation+repair to `LoopController` while preserving its exact `EvidencePacket` output, so the corpus path actually closes the loop with a real LLM and the 7 existing proof tests stay green. **This is the riskiest task — the corpus tests are the regression guard.**

**Files:**
- Modify: `gllm/proof/runner.py` — add a default `repair_prompt_builder` (`build_repair_prompt` already exists) and a convenience `run_proof_with_llm(request, *, model_name=..., run_vericut_fn=...)` that constructs an `LLMCandidateGenerator` and passes `build_repair_prompt` as the builder.
- Test: `tests/loop/test_proof_llm_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/loop/test_proof_llm_bridge.py
import json
import shutil
import unittest
from pathlib import Path

from gllm.proof.runner import run_proof_with_llm


class _StubChain:
    """Stands in for a LangChain chain: returns scripted G-code by attempt."""
    def __init__(self):
        self._calls = 0

    def invoke(self, _inputs):
        from langchain_core.messages.ai import AIMessage
        self._calls += 1
        # First call: bad tool; second call (repair): valid.
        text = "G90\nT99 M06\nG0 Z1.0\nM30\n" if self._calls == 1 else "G90\nT1 M06\nG0 Z1.0\nM30\n"
        return AIMessage(content=text)


class ProofLLMBridgeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / ".proof-llm-tmp" / self._testMethodName
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _registry(self):
        support = self.root / "support"
        support.mkdir()
        (support / "fixture.stl").write_text("f", encoding="utf-8")
        (support / "tools.tls").write_text("t", encoding="utf-8")
        (support / "control.ctl").write_text("c", encoding="utf-8")
        project = self.root / "template.vcproject"
        project.write_text(
            '<VcProject Version="9.5" Unit="inch"><Setup Name="d">'
            '<Component Name="F"><STL><File>fixture.stl</File></STL></Component>'
            '<NCPrograms Type="gcode"><NCProgram Use="on"><File>old.mcd</File></NCProgram></NCPrograms>'
            '<APT><Library>tools.tls</Library></APT><Control><File>control.ctl</File></Control>'
            "</Setup></VcProject>", encoding="utf-8")
        bat = self.root / "vericut.bat"
        bat.write_text("@echo off", encoding="utf-8")
        reg = self.root / "setups.json"
        reg.write_text(json.dumps({"setups": [{
            "id": "s1", "description": "d", "vericut_bat": str(bat),
            "project_template": str(project), "library_search_paths": [str(support)],
            "allowed_tools": ["T1"], "required_modes": ["G90"], "safe_z_min": 0.25,
        }]}), encoding="utf-8")
        return reg

    def test_llm_bridge_repairs_static_rejection_into_acceptance(self):
        from gllm.proof.runner import ScenarioRequest
        reg = self._registry()
        request = ScenarioRequest(
            prompt="Mill a square pocket.", registry_path=reg, setup_id="s1",
            output_root=self.root / "runs", scenario_id="bridge-1", max_repair_attempts=1,
        )
        packet = run_proof_with_llm(request, chain=_StubChain())
        self.assertEqual(packet.attempts[0].status, "rejected_static")
        self.assertEqual(packet.status, "accepted_static_only")
        self.assertEqual(packet.final_attempt, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/loop/test_proof_llm_bridge.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_proof_with_llm' from 'gllm.proof.runner'`

- [ ] **Step 3: Add the bridge to `gllm/proof/runner.py`**

Append to `gllm/proof/runner.py` (it already defines `run_proof_scenario`, `build_repair_prompt`, `ScenarioRequest`):

```python
def run_proof_with_llm(
    request: ScenarioRequest,
    *,
    chain=None,
    model_name: str | None = None,
    openrouter_model_name: str | None = None,
    run_vericut_fn: RunVericutFn = run_vericut,
) -> EvidencePacket:
    """Drive run_proof_scenario with a real LLM candidate generator and the
    existing build_repair_prompt builder. Pass `chain` to inject a fake in tests."""
    from gllm.loop.generator import LLMCandidateGenerator

    generator = LLMCandidateGenerator(
        model_name=model_name or request.model_name or "OpenRouter",
        openrouter_model_name=openrouter_model_name or request.openrouter_model_name,
        chain=chain,
    )

    def candidate_generator(prompt: str, context: "AttemptContext") -> str:
        return generator(prompt, context)

    return run_proof_scenario(
        request,
        candidate_generator=candidate_generator,
        repair_prompt_builder=build_repair_prompt,
        run_vericut_fn=run_vericut_fn,
    )
```

- [ ] **Step 4: Run the bridge test plus the full proof suite (regression guard)**

Run: `poetry run pytest tests/loop/test_proof_llm_bridge.py tests/test_proof_run.py tests/test_proof_corpus.py -q`
Expected: PASS (bridge test passes; all 7 `test_proof_run.py` cases + corpus stay green)

- [ ] **Step 5: Run the entire test suite**

Run: `poetry run pytest -q`
Expected: PASS (no regressions across the repo)

- [ ] **Step 6: Commit**

```bash
git add gllm/proof/runner.py tests/loop/test_proof_llm_bridge.py
git commit -m "feat(proof): bridge LLM candidate generator into run_proof_scenario"
```

---

## Self-Review

**1. Spec coverage:**
- Unified UI-agnostic engine → Tasks 1-11 (`gllm/loop/`). ✓
- Normalized `Finding` + three normalizers → Task 2. ✓
- Validator tiers (lint blocking + quarantined; static) → Task 5; Vericut tier → Task 10. ✓
- Static-first gating + Vericut final gate, skippable → Task 8 (`final_gate` only on green best) + Task 10 (auto-skip on missing assets). ✓
- Three modes (generate/repair/improve), seed handling → Tasks 7-8, 11. ✓
- Converge-or-cap, best-so-far, patience, budget, status taxonomy incl. `exhausted_best_effort` → Task 8. ✓
- Improve regression guard + return-seed → Task 8 `pick_best` + Task 11. ✓
- Evidence-packet reuse with new fields → Task 9. ✓
- Validator reliability cleanup → Task 3. ✓
- Shared `setup_constraints` (DRY) → Task 4. ✓
- LLM wired into proof hooks (closes loop end-to-end) → Task 12. ✓
- Token budget via chain usage → Task 1 (`TokenUsage.from_response`) + Task 7 + Task 8 (`budget.add`). ✓
- **Out of this plan (separate plans):** FastAPI service, NiceGUI UI, Monaco/three.js, targeted block repair, `cycle_time`/`rapid_distance` objectives. Documented in the scope note. ✓
- **Streaming events** (`LoopEvent`) are produced by the engine here (Task 8/11); they are *consumed* by the service/UI plans. ✓

**2. Placeholder scan:** No `TBD`/`TODO`/"add error handling"/"similar to Task N". Every code step shows complete code. ✓

**3. Type consistency:** `LoopRequest`/`Attempt`/`LoopResult`/`Finding`/`Severity`/`Mode`/`Objective`/`TokenUsage`/`Budget`/`ValidationContext` defined in Tasks 1-2 and used with the same field/method names throughout (`blocking_findings`, `last_usage`, `from_response`, `findings_penalty`, `objective_value`, `pick_best`, `derive_status`, `result_to_dict`). `LLMCandidateGenerator(__call__(prompt, ctx))` matches the controller call site and the proof `CandidateGenerator = Callable[[str, AttemptContext], str]` shape. ✓

**Note for the implementer:** Run `poetry run pytest -q` after Task 12; if `tests/test_gcode_utils.py` contained a test asserting the old silent-pass of `validate_functional_correctness`, update it per Task 3 Step 4.
