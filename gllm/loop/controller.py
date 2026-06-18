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
        if vericut.get("status") == "vericut_unavailable":
            # Vericut was requested and the candidate passed static gates, but the
            # final gate could not run. Static-clean but unverified by simulation.
            return "passed_vericut_unavailable"
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
            verdict = self._final_gate(best.gcode, ctx, request)
            # A None verdict means the gate was invoked but Vericut could not run
            # (no setup, missing assets/license, missing deps). Record that explicitly
            # so it is distinguishable from "Vericut was never requested".
            vericut = verdict if verdict is not None else {"status": "vericut_unavailable"}
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
