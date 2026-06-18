from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gllm.vericut.registry import load_setup_registry, validate_setup_paths
from gllm.vericut.runner import build_vericut_command, prepare_job_workspace, run_vericut
from gllm.vericut.static_checks import check_gcode
from gllm.vericut.verdict import write_verdict_packet


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare generated G-code for Vericut.")
    parser.add_argument("--registry", required=True, help="Path to a Vericut setup registry JSON.")
    parser.add_argument("--setup-id", required=True, help="Setup ID from the registry.")
    parser.add_argument("--gcode-file", required=True, help="Generated G-code file to stage.")
    parser.add_argument(
        "--output-root",
        default=".vericut-runs",
        help="Directory where prepared Vericut job folders are written.",
    )
    parser.add_argument("--job-id", default=None, help="Optional stable job ID.")
    parser.add_argument(
        "--skip-static-check",
        action="store_true",
        help="Prepare the job even if local static checks fail.",
    )
    parser.add_argument(
        "--run-vericut",
        action="store_true",
        help="After staging the job, invoke Vericut in BATCH mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Optional timeout for --run-vericut.",
    )
    args = parser.parse_args()

    registry = load_setup_registry(args.registry)
    setup = registry.get(args.setup_id)
    gcode = Path(args.gcode_file).read_text(encoding="utf-8")
    missing_paths = validate_setup_paths(setup)
    static_report = check_gcode(gcode, setup)

    if missing_paths:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "missing_setup_paths",
                    "missing_paths": missing_paths,
                },
                indent=2,
            )
        )
        return 2

    if not static_report.passed and not args.skip_static_check:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "static_check_failed",
                    "static_report": static_report.to_dict(),
                },
                indent=2,
            )
        )
        return 3

    job = prepare_job_workspace(
        setup=setup,
        gcode=gcode,
        output_root=args.output_root,
        job_id=args.job_id,
    )
    command = build_vericut_command(setup, job, batch=True)
    response = {
        "status": "prepared",
        "job_id": job.job_id,
        "workspace": str(job.workspace),
        "project_file": str(job.project_file),
        "generated_nc": str(job.generated_nc),
        "missing_dependencies": list(job.missing_dependencies),
        "static_report": static_report.to_dict(),
        "vericut_command": list(command),
    }

    if args.run_vericut:
        try:
            run_result = run_vericut(
                setup=setup,
                job=job,
                batch=True,
                timeout_seconds=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            response["status"] = "vericut_timeout"
            response["vericut_run"] = {
                "timeout_seconds": exc.timeout,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }
            print(json.dumps(response, indent=2))
            return 4

        response["status"] = "vericut_passed" if run_result.passed else "vericut_failed"
        response["vericut_run"] = {
            "returncode": run_result.returncode,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
            "artifacts": [str(path) for path in run_result.artifacts],
        }
        verdict = write_verdict_packet(
            job=job,
            static_report=static_report,
            run_result=run_result,
        )
        response["status"] = verdict.status
        response["vericut_verdict"] = verdict.to_dict()
        print(json.dumps(response, indent=2))
        return 0 if verdict.passed else run_result.returncode or 6

    print(
        json.dumps(response, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
