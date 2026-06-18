from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from gllm.vericut.runner import VericutJob, VericutRunResult
from gllm.vericut.static_checks import StaticCheckReport


@dataclass(frozen=True)
class VericutFinding:
    severity: str
    message: str
    line_number: int | None = None
    filename: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "message": self.message,
            "line_number": self.line_number,
            "filename": self.filename,
        }


@dataclass(frozen=True)
class ParsedVericutLog:
    error_count: int
    warning_count: int
    findings: tuple[VericutFinding, ...]
    cycle_time: str | None = None
    air_time: str | None = None
    air_time_percent: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "cycle_time": self.cycle_time,
            "air_time": self.air_time,
            "air_time_percent": self.air_time_percent,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class VericutArtifact:
    name: str
    path: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class VericutVerdict:
    status: str
    job_id: str
    passed: bool
    static_passed: bool
    process_returncode: int
    vericut: ParsedVericutLog
    artifacts: tuple[VericutArtifact, ...]
    verdict_file: Path
    summary_file: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "job_id": self.job_id,
            "passed": self.passed,
            "static_passed": self.static_passed,
            "process_returncode": self.process_returncode,
            "vericut": self.vericut.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "verdict_file": str(self.verdict_file),
            "summary_file": str(self.summary_file),
            "repair_context": _repair_context(self.vericut),
        }


_COUNT_RE = {
    "errors": re.compile(r"Number of Errors:\s*(\d+)", re.IGNORECASE),
    "warnings": re.compile(r"Number of Warnings:\s*(\d+)", re.IGNORECASE),
}
_CYCLE_TIME_RE = re.compile(r"Total cycle time since last rewind:\s*([0-9:]+)", re.IGNORECASE)
_AIR_TIME_RE = re.compile(
    r"Total air time since last rewind:\s*([0-9:]+)\s*\((\d+)\)%",
    re.IGNORECASE,
)
_FINDING_HEADER_RE = re.compile(
    r"^(Error|Warning) for line\s+(\d+),\s+Filename:\s*(.+)$",
    re.IGNORECASE,
)
_FINDING_MESSAGE_RE = re.compile(r"^(Error|Warning):\s*(.+)$", re.IGNORECASE)


def parse_vericut_log(log_text: str) -> ParsedVericutLog:
    findings = tuple(_extract_findings(log_text))
    error_count = _extract_count(log_text, "errors")
    warning_count = _extract_count(log_text, "warnings")

    if error_count is None:
        error_count = sum(1 for finding in findings if finding.severity == "error")
    if warning_count is None:
        warning_count = sum(1 for finding in findings if finding.severity == "warning")

    cycle_time_match = _CYCLE_TIME_RE.search(log_text)
    air_time_match = _AIR_TIME_RE.search(log_text)

    return ParsedVericutLog(
        error_count=error_count,
        warning_count=warning_count,
        findings=findings,
        cycle_time=cycle_time_match.group(1) if cycle_time_match else None,
        air_time=air_time_match.group(1) if air_time_match else None,
        air_time_percent=int(air_time_match.group(2)) if air_time_match else None,
    )


def write_verdict_packet(
    *,
    job: VericutJob,
    static_report: StaticCheckReport,
    run_result: VericutRunResult,
) -> VericutVerdict:
    artifacts = _collect_artifacts(job.output_dir)
    parsed_log = _parse_first_log(job, artifacts)
    status = _status_for(static_report, run_result, parsed_log, artifacts)
    verdict_file = job.output_dir / "verdict.json"
    summary_file = job.output_dir / "verdict.md"
    verdict = VericutVerdict(
        status=status,
        job_id=job.job_id,
        passed=status == "vericut_accepted",
        static_passed=static_report.passed,
        process_returncode=run_result.returncode,
        vericut=parsed_log,
        artifacts=artifacts,
        verdict_file=verdict_file,
        summary_file=summary_file,
    )

    verdict_file.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    summary_file.write_text(_markdown_summary(verdict), encoding="utf-8")
    return verdict


def _extract_count(log_text: str, key: str) -> int | None:
    match = _COUNT_RE[key].search(log_text)
    return int(match.group(1)) if match else None


def _extract_findings(log_text: str) -> list[VericutFinding]:
    lines = log_text.splitlines()
    findings: list[VericutFinding] = []

    for index, line in enumerate(lines):
        header_match = _FINDING_HEADER_RE.match(line.strip())
        if not header_match:
            continue

        message = ""
        for candidate in lines[index + 1 : index + 8]:
            message_match = _FINDING_MESSAGE_RE.match(candidate.strip())
            if message_match:
                message = message_match.group(2).strip()
                break

        findings.append(
            VericutFinding(
                severity=header_match.group(1).lower(),
                line_number=int(header_match.group(2)),
                filename=header_match.group(3).strip(),
                message=message or "Vericut reported an issue without a message.",
            )
        )

    return findings


def _collect_artifacts(output_dir: Path) -> tuple[VericutArtifact, ...]:
    if not output_dir.exists():
        return ()

    artifacts: list[VericutArtifact] = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name in {"verdict.json", "verdict.md"}:
            continue
        artifacts.append(
            VericutArtifact(
                name=path.name,
                path=str(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(artifacts)


def _parse_first_log(job: VericutJob, artifacts: tuple[VericutArtifact, ...]) -> ParsedVericutLog:
    for artifact in artifacts:
        if artifact.name.lower().endswith(".log"):
            return parse_vericut_log(Path(artifact.path).read_text(encoding="utf-8", errors="replace"))

    workspace_log = job.workspace / "vericut.log"
    if workspace_log.exists():
        return parse_vericut_log(workspace_log.read_text(encoding="utf-8", errors="replace"))

    return ParsedVericutLog(error_count=0, warning_count=0, findings=())


def _status_for(
    static_report: StaticCheckReport,
    run_result: VericutRunResult,
    parsed_log: ParsedVericutLog,
    artifacts: tuple[VericutArtifact, ...],
) -> str:
    has_log = any(artifact.name.lower().endswith(".log") for artifact in artifacts)
    # A rejection requires positive evidence that Vericut actually simulated the toolpath
    # and reported errors. Per policy this holds even when the process return code is 0.
    if has_log and parsed_log.error_count > 0:
        return "vericut_rejected"
    if (
        static_report.passed
        and run_result.returncode == 0
        and has_log
        and parsed_log.error_count == 0
    ):
        return "vericut_accepted"
    # No reported errors and not a clean accepted run: the simulation could not confirm
    # the toolpath — it failed to launch, produced no log, or exited abnormally without
    # reporting toolpath errors. That is "unverified", NOT a toolpath rejection.
    return "vericut_unverified"


def _markdown_summary(verdict: VericutVerdict) -> str:
    title_status = {
        "vericut_accepted": "Accepted",
        "vericut_rejected": "Rejected",
        "vericut_unverified": "Unverified",
    }.get(verdict.status, verdict.status)
    lines = [
        f"# Vericut Verdict: {title_status}",
        "",
        f"- Job: `{verdict.job_id}`",
        f"- Process return code: `{verdict.process_returncode}`",
        f"- Static checks passed: `{verdict.static_passed}`",
        f"- Vericut errors: `{verdict.vericut.error_count}`",
        f"- Vericut warnings: `{verdict.vericut.warning_count}`",
    ]
    if verdict.vericut.cycle_time:
        lines.append(f"- Cycle time: `{verdict.vericut.cycle_time}`")
    if verdict.vericut.air_time:
        lines.append(f"- Air time: `{verdict.vericut.air_time}`")
    if verdict.artifacts:
        lines.append(f"- Artifacts: `{', '.join(artifact.name for artifact in verdict.artifacts)}`")

    if verdict.vericut.findings:
        lines.extend(["", "## First Findings"])
        for finding in verdict.vericut.findings[:10]:
            line_label = f" line {finding.line_number}" if finding.line_number is not None else ""
            filename = f"{finding.filename}" if finding.filename else "unknown file"
            lines.append(f"- `{finding.severity}` in `{filename}`{line_label}: {finding.message}")

    return "\n".join(lines) + "\n"


def _repair_context(parsed_log: ParsedVericutLog) -> list[dict[str, object]]:
    return [finding.to_dict() for finding in parsed_log.findings[:10]]

