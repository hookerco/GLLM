from __future__ import annotations

from gllm.loop.types import Attempt, LoopResult, Mode

# Map engine statuses to operator actions, falling back to the proof runner's table.
_ACTION_OVERRIDES = {
    "passed": "ready_to_review",
    "passed_with_warnings": "ready_to_review",
    "passed_vericut_unavailable": "rerun_vericut",
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
