from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gllm.proof.runner import ScenarioRequest, run_proof_scenario


@dataclass(frozen=True)
class PromptCorpusScenario:
    id: str
    setup_id: str
    prompt: str
    candidate_gcode_file: Path
    expected_status: str
    tags: tuple[str, ...] = ()
    run_vericut: bool = False
    max_repair_attempts: int = 0


@dataclass(frozen=True)
class PromptCorpus:
    name: str
    scenarios: tuple[PromptCorpusScenario, ...]
    source_path: Path


@dataclass(frozen=True)
class PromptCorpusScenarioResult:
    scenario_id: str
    setup_id: str
    status: str
    expected_status: str
    operator_action: str
    packet_file: Path

    @property
    def passed(self) -> bool:
        return self.status == self.expected_status

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "setup_id": self.setup_id,
            "status": self.status,
            "expected_status": self.expected_status,
            "operator_action": self.operator_action,
            "passed": self.passed,
            "packet_file": str(self.packet_file),
        }


@dataclass(frozen=True)
class PromptCorpusResult:
    name: str
    results: tuple[PromptCorpusScenarioResult, ...]
    summary_file: Path

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def statuses(self) -> dict[str, str]:
        return {result.scenario_id: result.status for result in self.results}

    @property
    def setup_ids(self) -> tuple[str, ...]:
        return tuple(sorted({result.setup_id for result in self.results}))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": self.total,
            "setup_ids": list(self.setup_ids),
            "results": [result.to_dict() for result in self.results],
            "summary_file": str(self.summary_file),
        }


def load_prompt_corpus(path: str | Path) -> PromptCorpus:
    corpus_path = Path(path)
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    scenarios_payload = payload.get("scenarios")
    if not isinstance(scenarios_payload, list):
        raise ValueError("Prompt corpus must contain a 'scenarios' list")

    scenarios = tuple(
        _scenario_from_payload(entry, corpus_path.parent) for entry in scenarios_payload
    )
    return PromptCorpus(
        name=str(payload.get("name") or corpus_path.stem),
        scenarios=scenarios,
        source_path=corpus_path,
    )


def run_prompt_corpus(
    corpus: PromptCorpus,
    *,
    registry_path: str | Path,
    output_root: str | Path,
) -> PromptCorpusResult:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PromptCorpusScenarioResult] = []

    for scenario in corpus.scenarios:
        candidate_text = scenario.candidate_gcode_file.read_text(encoding="utf-8")
        request = ScenarioRequest(
            prompt=scenario.prompt,
            registry_path=registry_path,
            setup_id=scenario.setup_id,
            output_root=output_dir,
            scenario_id=scenario.id,
            model_name="CorpusFixture",
            prompt_type="Corpus candidate",
            run_vericut=scenario.run_vericut,
            max_repair_attempts=scenario.max_repair_attempts,
        )
        packet = run_proof_scenario(
            request,
            candidate_generator=lambda _prompt, _context, text=candidate_text: text,
        )
        results.append(
            PromptCorpusScenarioResult(
                scenario_id=scenario.id,
                setup_id=scenario.setup_id,
                status=packet.status,
                expected_status=scenario.expected_status,
                operator_action=packet.operator_action,
                packet_file=packet.packet_file,
            )
        )

    summary_file = output_dir / "corpus_summary.json"
    result = PromptCorpusResult(
        name=corpus.name,
        results=tuple(results),
        summary_file=summary_file,
    )
    summary_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    _write_markdown_summary(output_dir / "corpus_summary.md", result)
    return result


def _scenario_from_payload(
    payload: dict[str, object], base_dir: Path
) -> PromptCorpusScenario:
    candidate_path = Path(str(payload["candidate_gcode_file"]))
    if not candidate_path.is_absolute():
        candidate_path = base_dir / candidate_path
    return PromptCorpusScenario(
        id=str(payload["id"]),
        setup_id=str(payload["setup_id"]),
        prompt=str(payload["prompt"]),
        candidate_gcode_file=candidate_path,
        expected_status=str(payload.get("expected_status", "accepted_static_only")),
        tags=tuple(str(tag) for tag in payload.get("tags", ())),
        run_vericut=bool(payload.get("run_vericut", False)),
        max_repair_attempts=int(payload.get("max_repair_attempts", 0)),
    )


def _write_markdown_summary(path: Path, result: PromptCorpusResult) -> None:
    lines = [
        f"# Prompt Corpus: {result.name}",
        "",
        f"- Passed: `{result.passed}`",
        f"- Total scenarios: `{result.total}`",
        f"- Setups: `{', '.join(result.setup_ids)}`",
        "",
        "## Scenarios",
    ]
    for scenario_result in result.results:
        marker = "PASS" if scenario_result.passed else "FAIL"
        lines.append(
            f"- {marker} `{scenario_result.scenario_id}`: "
            f"`{scenario_result.status}` expected `{scenario_result.expected_status}` "
            f"({scenario_result.operator_action})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
