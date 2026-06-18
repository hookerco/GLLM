from __future__ import annotations

import contextlib
import io
import json
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from gllm.utils.gcode_utils import clean_gcode
from gllm.vericut.registry import load_setup_registry, validate_setup_paths
from gllm.vericut.runner import VericutJob, VericutRunResult, prepare_job_workspace, run_vericut
from gllm.vericut.static_checks import StaticCheckReport, check_gcode
from gllm.vericut.verdict import write_verdict_packet


CandidateGenerator = Callable[[str, "AttemptContext"], str]
RepairPromptBuilder = Callable[["AttemptContext"], str]
RunVericutFn = Callable[..., VericutRunResult]


@dataclass(frozen=True)
class ScenarioRequest:
    prompt: str
    registry_path: str | Path
    setup_id: str
    output_root: str | Path = ".proof-runs"
    scenario_id: str | None = None
    model_name: str = "unknown"
    prompt_type: str = "Unstructured"
    run_vericut: bool = False
    max_repair_attempts: int = 0
    timeout_seconds: int | None = None
    openrouter_model_name: str | None = None

    def resolved_scenario_id(self) -> str:
        return self.scenario_id or uuid.uuid4().hex

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "registry_path": str(self.registry_path),
            "setup_id": self.setup_id,
            "output_root": str(self.output_root),
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "prompt_type": self.prompt_type,
            "run_vericut": self.run_vericut,
            "max_repair_attempts": self.max_repair_attempts,
            "timeout_seconds": self.timeout_seconds,
            "openrouter_model_name": self.openrouter_model_name,
        }


@dataclass(frozen=True)
class AttemptContext:
    request: ScenarioRequest
    scenario_id: str
    attempt_number: int
    repair_context: tuple[dict[str, object], ...] = ()
    previous_gcode: str | None = None
    previous_status: str | None = None
    previous_raw_gcode_file: Path | None = None
    previous_cleaned_gcode_file: Path | None = None
    setup_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofAttempt:
    attempt_number: int
    status: str
    prompt_file: Path
    raw_gcode_file: Path
    cleaned_gcode_file: Path
    static_report: dict[str, object]
    job: dict[str, object] | None = None
    vericut: dict[str, object] | None = None
    plot_artifact: str | None = None
    error: str | None = None
    repair_context: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_number": self.attempt_number,
            "status": self.status,
            "operator_action": operator_action_for_status(self.status),
            "prompt_file": str(self.prompt_file),
            "raw_gcode_file": str(self.raw_gcode_file),
            "cleaned_gcode_file": str(self.cleaned_gcode_file),
            "static_report": self.static_report,
            "job": self.job,
            "vericut": self.vericut or {"status": "not_requested"},
            "plot_artifact": self.plot_artifact,
            "error": self.error,
            "repair_context": list(self.repair_context),
        }


@dataclass(frozen=True)
class EvidencePacket:
    scenario_id: str
    status: str
    request: ScenarioRequest
    attempts: tuple[ProofAttempt, ...]
    blockers: tuple[str, ...]
    repair_context: tuple[dict[str, object], ...]
    packet_file: Path
    summary_file: Path

    @property
    def final_attempt(self) -> int | None:
        if not self.attempts:
            return None
        return self.attempts[-1].attempt_number

    @property
    def operator_action(self) -> str:
        return operator_action_for_status(self.status)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "operator_action": self.operator_action,
            "request": self.request.to_dict() | {"scenario_id": self.scenario_id},
            "final_attempt": self.final_attempt,
            "blockers": list(self.blockers),
            "repair_context": list(self.repair_context),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "packet_file": str(self.packet_file),
            "summary_file": str(self.summary_file),
        }


def run_proof_scenario(
    request: ScenarioRequest,
    *,
    candidate_generator: CandidateGenerator,
    repair_prompt_builder: RepairPromptBuilder | None = None,
    run_vericut_fn: RunVericutFn = run_vericut,
) -> EvidencePacket:
    scenario_id = request.resolved_scenario_id()
    workspace = Path(request.output_root) / scenario_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "request.json").write_text(
        json.dumps(request.to_dict() | {"scenario_id": scenario_id}, indent=2),
        encoding="utf-8",
    )

    packet_file = workspace / "evidence_packet.json"
    summary_file = workspace / "evidence_packet.md"

    try:
        registry = load_setup_registry(request.registry_path)
        setup = registry.get(request.setup_id)
    except Exception as exc:
        return _write_packet(
            scenario_id=scenario_id,
            status="blocked_setup_registry",
            request=request,
            attempts=(),
            blockers=(str(exc),),
            repair_context=(),
            packet_file=packet_file,
            summary_file=summary_file,
        )

    missing_setup_paths = validate_setup_paths(setup)
    if missing_setup_paths:
        return _write_packet(
            scenario_id=scenario_id,
            status="blocked_setup_paths",
            request=request,
            attempts=(),
            blockers=tuple(missing_setup_paths),
            repair_context=(),
            packet_file=packet_file,
            summary_file=summary_file,
        )

    attempts: list[ProofAttempt] = []
    blockers: tuple[str, ...] = ()
    repair_context: tuple[dict[str, object], ...] = ()
    previous_gcode: str | None = None
    previous_status: str | None = None
    previous_raw_gcode_file: Path | None = None
    previous_cleaned_gcode_file: Path | None = None
    total_attempts = max(1, request.max_repair_attempts + 1)
    setup_constraints = _setup_constraints(setup)

    for attempt_number in range(1, total_attempts + 1):
        context = AttemptContext(
            request=request,
            scenario_id=scenario_id,
            attempt_number=attempt_number,
            repair_context=repair_context,
            previous_gcode=previous_gcode,
            previous_status=previous_status,
            previous_raw_gcode_file=previous_raw_gcode_file,
            previous_cleaned_gcode_file=previous_cleaned_gcode_file,
            setup_constraints=setup_constraints,
        )
        if attempt_number == 1:
            prompt = request.prompt
        elif repair_prompt_builder is None:
            break
        else:
            prompt = repair_prompt_builder(context)

        attempt_dir = workspace / f"attempt-{attempt_number:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = attempt_dir / "prompt.txt"
        raw_gcode_file = attempt_dir / "candidate.raw.txt"
        cleaned_gcode_file = attempt_dir / "candidate.nc"
        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            raw_gcode = candidate_generator(prompt, context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw_gcode_file.write_text("", encoding="utf-8")
            cleaned_gcode_file.write_text("", encoding="utf-8")
            blockers = (error,)
            attempts.append(
                ProofAttempt(
                    attempt_number=attempt_number,
                    status="blocked_generation_failed",
                    prompt_file=prompt_file,
                    raw_gcode_file=raw_gcode_file,
                    cleaned_gcode_file=cleaned_gcode_file,
                    static_report={"passed": False, "findings": []},
                    error=traceback.format_exc(),
                )
            )
            break

        cleaned_gcode = clean_gcode(raw_gcode)
        raw_gcode_file.write_text(raw_gcode, encoding="utf-8")
        cleaned_gcode_file.write_text(cleaned_gcode, encoding="utf-8")
        static_report = check_gcode(cleaned_gcode, setup)
        plot_artifact = _write_gcode_plot(cleaned_gcode, attempt_dir)

        if not static_report.passed:
            repair_context = _static_repair_context(static_report)
            attempt = ProofAttempt(
                attempt_number=attempt_number,
                status="rejected_static",
                prompt_file=prompt_file,
                raw_gcode_file=raw_gcode_file,
                cleaned_gcode_file=cleaned_gcode_file,
                static_report=static_report.to_dict(),
                vericut={"status": "skipped", "reason": "static_checks_failed"},
                plot_artifact=plot_artifact,
                repair_context=repair_context,
            )
            attempts.append(attempt)
            previous_gcode = cleaned_gcode
            previous_status = attempt.status
            previous_raw_gcode_file = raw_gcode_file
            previous_cleaned_gcode_file = cleaned_gcode_file
            continue

        try:
            job = prepare_job_workspace(
                setup=setup,
                gcode=cleaned_gcode,
                output_root=workspace / "vericut-jobs",
                job_id=f"attempt-{attempt_number:03d}",
            )
        except Exception as exc:
            blockers = (f"{type(exc).__name__}: {exc}",)
            attempts.append(
                ProofAttempt(
                    attempt_number=attempt_number,
                    status="blocked_staging_failed",
                    prompt_file=prompt_file,
                    raw_gcode_file=raw_gcode_file,
                    cleaned_gcode_file=cleaned_gcode_file,
                    static_report=static_report.to_dict(),
                    vericut={"status": "skipped", "reason": "staging_failed"},
                    plot_artifact=plot_artifact,
                    error=traceback.format_exc(),
                )
            )
            break

        job_payload = _job_to_dict(job)
        if job.missing_dependencies:
            blockers = tuple(job.missing_dependencies)
            attempts.append(
                ProofAttempt(
                    attempt_number=attempt_number,
                    status="blocked_missing_dependencies",
                    prompt_file=prompt_file,
                    raw_gcode_file=raw_gcode_file,
                    cleaned_gcode_file=cleaned_gcode_file,
                    static_report=static_report.to_dict(),
                    job=job_payload,
                    vericut={"status": "skipped", "reason": "missing_dependencies"},
                    plot_artifact=plot_artifact,
                )
            )
            break

        if not request.run_vericut:
            attempts.append(
                ProofAttempt(
                    attempt_number=attempt_number,
                    status="accepted_static_only",
                    prompt_file=prompt_file,
                    raw_gcode_file=raw_gcode_file,
                    cleaned_gcode_file=cleaned_gcode_file,
                    static_report=static_report.to_dict(),
                    job=job_payload,
                    vericut={"status": "skipped", "reason": "run_vericut_false"},
                    plot_artifact=plot_artifact,
                )
            )
            repair_context = ()
            break

        try:
            run_result = run_vericut_fn(
                setup,
                job,
                batch=True,
                timeout_seconds=request.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            blockers = (f"Vericut timed out after {exc.timeout} seconds.",)
            attempts.append(
                ProofAttempt(
                    attempt_number=attempt_number,
                    status="blocked_vericut_timeout",
                    prompt_file=prompt_file,
                    raw_gcode_file=raw_gcode_file,
                    cleaned_gcode_file=cleaned_gcode_file,
                    static_report=static_report.to_dict(),
                    job=job_payload,
                    vericut={
                        "status": "timeout",
                        "timeout_seconds": exc.timeout,
                        "stdout": exc.stdout or "",
                        "stderr": exc.stderr or "",
                    },
                    plot_artifact=plot_artifact,
                )
            )
            break

        verdict = write_verdict_packet(
            job=job,
            static_report=static_report,
            run_result=run_result,
        )
        verdict_payload = verdict.to_dict()
        repair_context = tuple(verdict_payload.get("repair_context", ()))
        status = _status_for_vericut_verdict(verdict_payload)
        attempt = ProofAttempt(
            attempt_number=attempt_number,
            status=status,
            prompt_file=prompt_file,
            raw_gcode_file=raw_gcode_file,
            cleaned_gcode_file=cleaned_gcode_file,
            static_report=static_report.to_dict(),
            job=job_payload,
            vericut=verdict_payload,
            plot_artifact=plot_artifact,
            repair_context=repair_context,
        )
        attempts.append(attempt)
        previous_gcode = cleaned_gcode
        previous_status = attempt.status
        previous_raw_gcode_file = raw_gcode_file
        previous_cleaned_gcode_file = cleaned_gcode_file
        if verdict.passed:
            repair_context = ()
            break

    final_status = _final_status(attempts, blockers)
    return _write_packet(
        scenario_id=scenario_id,
        status=final_status,
        request=request,
        attempts=tuple(attempts),
        blockers=blockers,
        repair_context=repair_context,
        packet_file=packet_file,
        summary_file=summary_file,
    )


def build_repair_prompt(context: AttemptContext) -> str:
    findings = _format_repair_context(context.repair_context)
    previous_gcode = context.previous_gcode or ""
    constraints = "\n".join(f"- {constraint}" for constraint in context.setup_constraints)
    if not constraints:
        constraints = "- No setup constraints were provided."
    previous_candidate_file = context.previous_cleaned_gcode_file or context.previous_raw_gcode_file
    previous_candidate_line = (
        f"Previous candidate file: {previous_candidate_file}\n\n"
        if previous_candidate_file is not None
        else ""
    )
    return (
        "The previous generated G-code was rejected by local machine checks. "
        "Repair it using only the concrete findings below. Return the full corrected "
        "G-code program and no prose.\n\n"
        f"Scenario: {context.scenario_id}\n"
        f"Repair attempt: {context.attempt_number}\n"
        f"Setup ID: {context.request.setup_id}\n"
        f"{previous_candidate_line}"
        f"Original task:\n{context.request.prompt}\n\n"
        f"Previous status: {context.previous_status}\n\n"
        f"Setup constraints:\n{constraints}\n\n"
        f"Findings:\n{findings}\n\n"
        f"Previous G-code:\n{previous_gcode}\n"
    )


def operator_action_for_status(status: str) -> str:
    if status == "accepted_vericut":
        return "ready_to_review"
    if status == "accepted_static_only":
        return "rerun_vericut"
    if status in {"rejected_static", "rejected_vericut", "blocked_generation_failed"}:
        return "fix_prompt"
    if status in {
        "blocked_setup_registry",
        "blocked_setup_paths",
        "blocked_missing_dependencies",
        "blocked_staging_failed",
    }:
        return "fix_setup"
    if status == "blocked_vericut_timeout":
        return "rerun_vericut"
    if status == "blocked_vericut_unverified":
        return "rerun_vericut"
    if status == "reject":
        return "reject"
    return "manual_review_required"


def _status_for_vericut_verdict(verdict_payload: dict[str, object]) -> str:
    if verdict_payload.get("passed") is True:
        return "accepted_vericut"
    if verdict_payload.get("status") == "vericut_unverified":
        return "blocked_vericut_unverified"
    return "rejected_vericut"


def _write_packet(
    *,
    scenario_id: str,
    status: str,
    request: ScenarioRequest,
    attempts: tuple[ProofAttempt, ...],
    blockers: tuple[str, ...],
    repair_context: tuple[dict[str, object], ...],
    packet_file: Path,
    summary_file: Path,
) -> EvidencePacket:
    packet = EvidencePacket(
        scenario_id=scenario_id,
        status=status,
        request=request,
        attempts=attempts,
        blockers=blockers,
        repair_context=repair_context,
        packet_file=packet_file,
        summary_file=summary_file,
    )
    packet_file.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")
    summary_file.write_text(_packet_summary(packet), encoding="utf-8")
    return packet


def _final_status(attempts: list[ProofAttempt], blockers: tuple[str, ...]) -> str:
    if blockers and attempts:
        return attempts[-1].status
    if blockers:
        return "blocked"
    if not attempts:
        return "blocked_no_attempts"
    return attempts[-1].status


def _static_repair_context(static_report: StaticCheckReport) -> tuple[dict[str, object], ...]:
    return tuple(finding.to_dict() for finding in static_report.findings)


def _setup_constraints(setup) -> tuple[str, ...]:
    from gllm.loop.constraints import setup_constraints

    return setup_constraints(setup)


def _job_to_dict(job: VericutJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "workspace": str(job.workspace),
        "input_dir": str(job.input_dir),
        "output_dir": str(job.output_dir),
        "request_file": str(job.request_file),
        "generated_nc": str(job.generated_nc),
        "project_file": str(job.project_file),
        "copied_dependencies": [str(path) for path in job.copied_dependencies],
        "missing_dependencies": list(job.missing_dependencies),
    }


def _write_gcode_plot(gcode: str, attempt_dir: Path) -> str | None:
    if not gcode.strip():
        return None
    try:
        from gllm.utils.plot_utils import plot_gcode

        plot_path = attempt_dir / "gcode_path.png"
        with contextlib.redirect_stdout(io.StringIO()):
            plt = plot_gcode(gcode)
        plt.savefig(plot_path)
        plt.close("all")
        return str(plot_path)
    except Exception:
        return None


def _format_repair_context(repair_context: Iterable[dict[str, object]]) -> str:
    lines: list[str] = []
    for finding in repair_context:
        code = finding.get("code") or finding.get("severity") or "finding"
        line_number = finding.get("line_number")
        line_label = f" line {line_number}" if line_number is not None else ""
        message = finding.get("message", "")
        lines.append(f"- {code}{line_label}: {message}")
    return "\n".join(lines) if lines else "- No concrete findings were reported."


def _packet_summary(packet: EvidencePacket) -> str:
    lines = [
        f"# Proof Run: {packet.status}",
        "",
        f"- Scenario: `{packet.scenario_id}`",
        f"- Setup: `{packet.request.setup_id}`",
        f"- Model: `{packet.request.model_name}`",
        f"- Prompt type: `{packet.request.prompt_type}`",
        f"- Operator action: `{packet.operator_action}`",
        f"- Attempts: `{len(packet.attempts)}`",
    ]
    if packet.blockers:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {blocker}" for blocker in packet.blockers)
    if packet.attempts:
        lines.extend(["", "## Attempts"])
        for attempt in packet.attempts:
            lines.append(f"- Attempt {attempt.attempt_number}: `{attempt.status}`")
    if packet.repair_context:
        lines.extend(["", "## Repair Context"])
        lines.extend(_format_repair_context(packet.repair_context).splitlines())
    return "\n".join(lines) + "\n"
