from __future__ import annotations

import argparse
import json
from pathlib import Path

from gllm.proof.runner import ScenarioRequest, build_repair_prompt, run_proof_scenario
from gllm.utils.gcode_utils import clean_gcode
from gllm.utils.model_utils import DEFAULT_MODEL, MODEL_OPTIONS, setup_langchain_without_rag, setup_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a prompt-to-verdict proof packet.")
    parser.add_argument("--registry", required=True, help="Path to a Vericut setup registry JSON.")
    parser.add_argument("--setup-id", required=True, help="Setup ID from the registry.")
    parser.add_argument("--prompt", help="Natural-language CNC task.")
    parser.add_argument("--prompt-file", help="File containing the natural-language CNC task.")
    parser.add_argument(
        "--candidate-gcode-file",
        help="Use an existing candidate G-code file instead of calling a model.",
    )
    parser.add_argument(
        "--output-root",
        default=".proof-runs",
        help="Directory where proof run folders are written.",
    )
    parser.add_argument("--scenario-id", default=None, help="Optional stable scenario ID.")
    parser.add_argument("--model", choices=MODEL_OPTIONS, default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-model", default=None, help="Optional OpenRouter model ID.")
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=0,
        help="Maximum repair attempts after the first rejected candidate.",
    )
    parser.add_argument(
        "--run-vericut",
        action="store_true",
        help="Invoke Vericut after staging. This may consume a local Vericut license.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    prompt = _read_prompt(args.prompt, args.prompt_file)
    request = ScenarioRequest(
        prompt=prompt,
        registry_path=args.registry,
        setup_id=args.setup_id,
        output_root=args.output_root,
        scenario_id=args.scenario_id,
        model_name=args.model,
        prompt_type="Unstructured",
        run_vericut=args.run_vericut,
        max_repair_attempts=max(args.max_repair_attempts, 0),
        timeout_seconds=args.timeout_seconds,
        openrouter_model_name=args.openrouter_model,
    )

    if args.candidate_gcode_file:
        candidate_text = Path(args.candidate_gcode_file).read_text(encoding="utf-8")

        def candidate_generator(generation_prompt, context):
            return candidate_text

        repair_prompt_builder = None
    else:
        model = setup_model(args.model, openrouter_model_name=args.openrouter_model)
        chain = setup_langchain_without_rag(model=model)

        def candidate_generator(generation_prompt, context):
            chain_prompt = (
                _generation_prompt(generation_prompt)
                if context.attempt_number == 1
                else generation_prompt
            )
            response = chain.invoke({"input": chain_prompt})
            return clean_gcode(response.content if hasattr(response, "content") else str(response))

        repair_prompt_builder = build_repair_prompt

    packet = run_proof_scenario(
        request,
        candidate_generator=candidate_generator,
        repair_prompt_builder=repair_prompt_builder,
    )
    print(json.dumps(packet.to_dict(), indent=2))
    return 0 if packet.status.startswith("accepted") else 2


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt:
        return prompt
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --prompt or --prompt-file.")


def _generation_prompt(task_description: str) -> str:
    return (
        "Based on the details provided, generate a robust G-code for the CNC "
        "machining operation. Return the complete G-code program and no prose.\n\n"
        f"Task description: {task_description}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
