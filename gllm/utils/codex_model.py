"""
LangChain-adjacent wrapper for using Codex CLI auth as a model backend.

Codex OAuth/access tokens are consumed by `codex exec`; this module never reads
or serializes token values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class CodexCliLanguageModel:
    command: str = "codex"
    timeout_seconds: int = 180
    cwd: Optional[str] = None
    runner: Runner = field(default=subprocess.run, repr=False)
    extra_args: Sequence[str] = field(default_factory=tuple)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        prompt = self._format_input(input)
        args = [
            self.command,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            *self.extra_args,
            "-",
        ]
        completed = self.runner(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=self.cwd,
        )
        if completed.returncode != 0:
            detail = self._safe_failure_detail(completed.stderr)
            raise RuntimeError(f"Codex CLI model failed: {detail}")
        return completed.stdout.strip()

    @staticmethod
    def _format_input(input: Any) -> str:
        if isinstance(input, Mapping):
            if "input" in input:
                return str(input["input"])
            if "question" in input:
                return str(input["question"])
        if hasattr(input, "to_string"):
            return input.to_string()
        if hasattr(input, "content"):
            return str(input.content)
        return str(input)

    @staticmethod
    def _safe_failure_detail(stderr: str) -> str:
        text = (stderr or "").strip()
        if not text:
            return "no stderr was returned"
        lowered = text.lower()
        if "token" in lowered or "api key" in lowered or "credential" in lowered:
            return "authentication failed or credentials are unavailable"
        return text[-500:]


class CodexPromptChain:
    def __init__(self, model: CodexCliLanguageModel, system_template: str):
        self.model = model
        self.system_template = system_template

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        user_input = CodexCliLanguageModel._format_input(input)
        return self.model.invoke(self.system_template.format(input=user_input))
