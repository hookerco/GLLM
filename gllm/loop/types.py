from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from gllm.loop.findings import Finding


class Mode(str, Enum):
    GENERATE = "generate"
    REPAIR = "repair"
    IMPROVE = "improve"


@dataclass(frozen=True)
class Objective:
    metric: Literal["path_length", "tool_changes"]
    direction: Literal["minimize"] = "minimize"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, response) -> "TokenUsage":
        meta = getattr(response, "usage_metadata", None) or {}
        return cls(
            int(meta.get("input_tokens", 0) or 0),
            int(meta.get("output_tokens", 0) or 0),
        )


@dataclass
class Budget:
    max_attempts: int = 4
    token_budget: int | None = None
    patience: int = 2
    spent_tokens: int = 0

    def add(self, usage: "TokenUsage") -> None:
        self.spent_tokens += usage.total

    def exhausted(self) -> bool:
        return self.token_budget is not None and self.spent_tokens >= self.token_budget


@dataclass(frozen=True)
class ValidationContext:
    setup: object | None = None          # gllm.vericut.registry.VericutSetup | None
    feed_rate_min: float = 1.0
    feed_rate_max: float = 100.0
    spindle_speed_max: float = 900.0

    @classmethod
    def from_setup(cls, setup) -> "ValidationContext":
        if setup is None:
            return cls()
        return cls(
            setup=setup,
            feed_rate_min=setup.feed_rate_min if setup.feed_rate_min is not None else 1.0,
            feed_rate_max=setup.feed_rate_max if setup.feed_rate_max is not None else 100.0,
            spindle_speed_max=(
                setup.spindle_speed_max if setup.spindle_speed_max is not None else 900.0
            ),
        )


@dataclass(frozen=True)
class LoopRequest:
    prompt: str
    mode: Mode = Mode.GENERATE
    registry_path: str | Path | None = None
    setup_id: str | None = None
    seed_gcode: str | None = None
    objective: Objective | None = None
    model_name: str = "OpenRouter"
    openrouter_model_name: str | None = None
    run_vericut: bool = False
    max_attempts: int = 4
    token_budget: int | None = None
    patience: int = 2
    output_root: str | Path = ".loop-runs"
    scenario_id: str | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class Attempt:
    number: int
    gcode: str
    findings: tuple["Finding", ...]
    score: float
    objective_value: float | None = None

    @property
    def blocking_findings(self) -> tuple["Finding", ...]:
        return tuple(f for f in self.findings if f.blocking)


@dataclass(frozen=True)
class LoopResult:
    request: LoopRequest
    scenario_id: str
    status: str
    best: Attempt | None
    history: tuple[Attempt, ...]
    vericut: dict | None = None


@dataclass(frozen=True)
class LoopEvent:
    type: str
    attempt: Attempt | None = None
    prompt: str | None = None
    result: LoopResult | None = None
    payload: dict | None = None
