from __future__ import annotations

from typing import Iterable, Protocol

from langchain_core.messages.ai import AIMessage

from gllm.loop.findings import Finding, format_findings
from gllm.loop.types import LoopRequest, Objective, TokenUsage
from gllm.utils.model_utils import setup_langchain_without_rag, setup_model


class CandidateGenerator(Protocol):
    last_usage: TokenUsage

    def __call__(self, prompt: str, ctx: object | None = None) -> str: ...


class LLMCandidateGenerator:
    """Pure prompt -> raw G-code text. Builds its chain lazily from model_utils."""

    def __init__(self, model_name: str = "OpenRouter", openrouter_model_name: str | None = None, chain=None):
        self._model_name = model_name
        self._openrouter_model_name = openrouter_model_name
        self._chain = chain
        self.last_usage = TokenUsage()

    def _ensure_chain(self):
        if self._chain is None:
            model = setup_model(self._model_name, self._openrouter_model_name)
            self._chain = setup_langchain_without_rag(model)
        return self._chain

    def __call__(self, prompt: str, ctx: object | None = None) -> str:
        chain = self._ensure_chain()
        response = chain.invoke({"input": prompt})
        self.last_usage = TokenUsage.from_response(response)
        return response.content if isinstance(response, AIMessage) else str(response)


def _constraints_block(constraints: Iterable[str]) -> str:
    items = list(constraints)
    return "\n".join(f"- {c}" for c in items) if items else "- No machine constraints were provided."


def build_generate_prompt(request: LoopRequest, constraints: Iterable[str]) -> str:
    return (
        "Generate a complete, robust CNC G-code program for the task below. "
        "Return only G-code and no prose.\n\n"
        f"Task:\n{request.prompt}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n"
    )


def build_repair_prompt(
    request: LoopRequest,
    previous_gcode: str,
    findings: Iterable[Finding],
    constraints: Iterable[str],
) -> str:
    return (
        "The previous G-code was rejected by machine checks. Repair it using only the "
        "concrete findings below. Return the full corrected G-code program and no prose.\n\n"
        f"Original task:\n{request.prompt}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n\n"
        f"Findings:\n{format_findings(findings)}\n\n"
        f"Previous G-code:\n{previous_gcode}\n"
    )


def build_improve_prompt(
    request: LoopRequest,
    current_gcode: str,
    objective: Objective,
    current_value: float | None,
    constraints: Iterable[str],
) -> str:
    value_text = "unknown" if current_value is None else f"{current_value:g}"
    return (
        "The G-code below already passes all machine checks. Improve it to "
        f"{objective.direction} {objective.metric} WITHOUT violating any constraint. "
        "Every machine check must still pass. Return the full improved G-code program and no prose.\n\n"
        f"Original task:\n{request.prompt}\n\n"
        f"Current {objective.metric}: {value_text}\n\n"
        f"Machine constraints:\n{_constraints_block(constraints)}\n\n"
        f"Current G-code:\n{current_gcode}\n"
    )
