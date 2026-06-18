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
