from __future__ import annotations

import uuid
from typing import Callable, Iterator, Sequence

from gllm.loop.constraints import setup_constraints
from gllm.loop.findings import Finding, from_vericut
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


def _vericut_error_count(verdict: dict) -> int:
    vericut = verdict.get("vericut") or {}
    count = vericut.get("error_count")
    return count if isinstance(count, int) else 0


def _pick_best_vericut(best: tuple, candidate: tuple) -> tuple:
    """Choose the better (Attempt, verdict) pair across Vericut repair rounds:
    an accepted verdict beats a non-accepted one; among rejected, fewer Vericut
    errors wins; ties keep the later (candidate) pair."""
    if candidate[1].get("passed") is True:
        return candidate
    if best[1].get("passed") is True:
        return best
    return candidate if _vericut_error_count(candidate[1]) <= _vericut_error_count(best[1]) else best


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

            # A toolpath rejection carries concrete findings; feed them back and
            # re-simulate, up to a bounded number of rounds (or until accepted/budget).
            best, vericut = yield from self._repair_after_vericut(
                request, ctx, constraints, best, vericut, history, budget
            )

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

    def _repair_after_vericut(self, request, ctx, constraints, best, verdict, history, budget):
        """Run up to ``vericut_max_rounds`` repair rounds on a Vericut rejection.

        Each round feeds the reported Vericut findings (plus any static findings a
        prior repaired candidate regressed on) into a fresh candidate, re-validates it
        statically, and re-simulates it only when it is statically clean. Yields the
        per-round LoopEvents and returns the best ``(Attempt, verdict)`` pair seen."""
        best_pair = (best, verdict)
        current_gcode = best.gcode
        carried_static: list[Finding] = []
        rounds = 0
        while (
            verdict.get("status") == "vericut_rejected"
            and rounds < request.vericut_max_rounds
            and not budget.exhausted()
        ):
            rounds += 1
            vfindings = [from_vericut(p) for p in verdict.get("repair_context", [])]
            prompt = build_repair_prompt(request, current_gcode, vfindings + carried_static, constraints)
            yield LoopEvent("repair_prompt", prompt=prompt)
            candidate = self._generator(prompt, ctx)
            budget.add(self._generator.last_usage)

            cleaned = clean_gcode(candidate)
            findings = self._validate(cleaned, ctx)
            attempt = Attempt(
                len(history) + 1,
                cleaned,
                tuple(findings),
                findings_penalty(findings),
                objective_value(cleaned, request.objective),
            )
            history.append(attempt)
            yield LoopEvent("findings", attempt=attempt)
            current_gcode = attempt.gcode

            if attempt.blocking_findings:
                # The repaired candidate regressed on the static gates. Don't spend a
                # Vericut run on it; carry its findings into the next repair prompt.
                carried_static = list(attempt.findings)
                continue

            carried_static = []
            yield LoopEvent("vericut_started")
            verdict = self._final_gate(attempt.gcode, ctx, request) or {"status": "vericut_unavailable"}
            yield LoopEvent("vericut_verdict", payload=verdict)
            best_pair = _pick_best_vericut(best_pair, (attempt, verdict))

        return best_pair

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
