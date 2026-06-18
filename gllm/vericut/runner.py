from __future__ import annotations

import json
import shutil
import subprocess
import stat
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gllm.vericut.registry import VericutSetup


@dataclass(frozen=True)
class VericutJob:
    job_id: str
    workspace: Path
    input_dir: Path
    output_dir: Path
    request_file: Path
    generated_nc: Path
    project_file: Path
    copied_dependencies: tuple[Path, ...]
    missing_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class VericutRunResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    artifacts: tuple[Path, ...] = ()

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def prepare_job_workspace(
    setup: VericutSetup,
    gcode: str,
    output_root: str | Path,
    job_id: str | None = None,
) -> VericutJob:
    job_id = job_id or uuid.uuid4().hex
    workspace = Path(output_root) / job_id
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_nc = input_dir / "generated.nc"
    generated_nc.write_text(gcode, encoding="utf-8")

    tree = ET.parse(setup.project_template)
    root = tree.getroot()
    references = _collect_project_references(root)
    _replace_nc_programs(root, generated_nc.name)

    project_file = input_dir / setup.project_template.name
    tree.write(project_file, encoding="utf-8", xml_declaration=True)

    copied_dependencies, missing_dependencies = _copy_dependencies(
        references.dependencies,
        input_dir=input_dir,
        project_template=setup.project_template,
        search_paths=setup.library_search_paths,
    )
    request_file = workspace / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "setup_id": setup.id,
                "job_id": job_id,
                "source_project_template": str(setup.project_template),
                "project_file": str(project_file),
                "generated_nc": str(generated_nc),
                "copied_dependencies": [str(path) for path in copied_dependencies],
                "missing_dependencies": list(missing_dependencies),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return VericutJob(
        job_id=job_id,
        workspace=workspace,
        input_dir=input_dir,
        output_dir=output_dir,
        request_file=request_file,
        generated_nc=generated_nc,
        project_file=project_file,
        copied_dependencies=tuple(copied_dependencies),
        missing_dependencies=tuple(missing_dependencies),
    )


def build_vericut_command(
    setup: VericutSetup,
    job: VericutJob,
    *,
    batch: bool = True,
) -> tuple[str, ...]:
    command = [str(setup.vericut_bat)]
    if batch:
        command.append("BATCH")
    command.append(str(job.project_file.resolve()))
    return tuple(command)


def run_vericut(
    setup: VericutSetup,
    job: VericutJob,
    *,
    batch: bool = True,
    timeout_seconds: int | None = None,
) -> VericutRunResult:
    command = build_vericut_command(setup, job, batch=batch)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=job.workspace.resolve(),
    )
    artifacts = _collect_run_outputs(job)
    return VericutRunResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        artifacts=artifacts,
    )


@dataclass(frozen=True)
class _ProjectReferences:
    nc_programs: tuple[str, ...]
    dependencies: tuple[str, ...]


def _collect_project_references(root: ET.Element) -> _ProjectReferences:
    nc_programs: list[str] = []
    dependencies: list[str] = []

    def walk(element: ET.Element, ancestors: tuple[str, ...]) -> None:
        tag = _local_name(element.tag)
        next_ancestors = ancestors + (tag,)
        text = (element.text or "").strip()
        if tag in {"File", "Library"} and text and not _is_ignored_reference(text, ancestors):
            if "NCPrograms" in ancestors or "NCProgram" in ancestors:
                nc_programs.append(text)
            else:
                dependencies.append(text)
        for child in element:
            walk(child, next_ancestors)

    walk(root, ())
    return _ProjectReferences(
        nc_programs=tuple(_dedupe(nc_programs)),
        dependencies=tuple(_dedupe(dependencies)),
    )


def _replace_nc_programs(root: ET.Element, program_name: str) -> None:
    def walk(element: ET.Element, inside_nc_programs: bool) -> None:
        tag = _local_name(element.tag)
        child_inside_nc_programs = inside_nc_programs or tag == "NCPrograms"
        if tag == "File" and inside_nc_programs:
            element.text = program_name
        for child in element:
            walk(child, child_inside_nc_programs)

    walk(root, False)


def _copy_dependencies(
    references: Iterable[str],
    *,
    input_dir: Path,
    project_template: Path,
    search_paths: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    copied: list[Path] = []
    missing: list[str] = []
    seen_destinations: set[Path] = set()
    template_dir = project_template.parent

    for reference in references:
        resolved = _resolve_dependency(reference, template_dir, search_paths)
        if resolved is None:
            missing.append(reference)
            continue

        destination = input_dir / resolved.name
        if destination in seen_destinations:
            continue
        seen_destinations.add(destination)
        if resolved.resolve() != destination.resolve():
            _copy_over_existing(resolved, destination)
        copied.append(destination)

    return tuple(copied), tuple(missing)


def _resolve_dependency(
    reference: str,
    template_dir: Path,
    search_paths: tuple[Path, ...],
) -> Path | None:
    candidate = Path(reference)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(template_dir / candidate)
        candidates.extend(path / candidate for path in search_paths)
    if candidate.name != str(candidate):
        candidates.append(template_dir / candidate.name)
        candidates.extend(path / candidate.name for path in search_paths)

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _copy_over_existing(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.chmod(destination.stat().st_mode | stat.S_IWRITE)
        destination.unlink()
    shutil.copy2(source, destination)


def _collect_run_outputs(job: VericutJob) -> tuple[Path, ...]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    for pattern in ("vericut.log", "results.htm", "*_metrics.csv", "vericut*.jpg", "vericut*.jpeg", "vericut*.png"):
        for source in sorted(job.workspace.glob(pattern)):
            if not source.is_file():
                continue
            destination = job.output_dir / source.name
            if source.resolve() != destination.resolve():
                _copy_over_existing(source, destination)
                source.unlink()
            collected.append(destination)
    return tuple(collected)


def _is_ignored_reference(reference: str, ancestors: tuple[str, ...]) -> bool:
    output_contexts = {"ProjectReport", "Report", "MetricsDelim", "Picture", "Export", "STLOutput"}
    if output_contexts.intersection(ancestors):
        return True
    if any(char in reference for char in "*?"):
        return True
    return reference.strip().lower().startswith(("http://", "https://"))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
