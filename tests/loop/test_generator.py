import unittest
from langchain_core.messages.ai import AIMessage
from gllm.loop.generator import (
    LLMCandidateGenerator, build_generate_prompt, build_repair_prompt, build_improve_prompt,
)
from gllm.loop.types import LoopRequest, Objective
from gllm.loop.findings import Finding, Severity


class _FakeChain:
    def __init__(self, content, usage=None):
        self._content = content
        self._usage = usage

    def invoke(self, _inputs):
        msg = AIMessage(content=self._content)
        if self._usage is not None:
            msg.usage_metadata = self._usage
        return msg


class GeneratorTests(unittest.TestCase):
    def test_generator_returns_text_and_tracks_usage(self):
        chain = _FakeChain("G90\nM30", usage={"input_tokens": 10, "output_tokens": 4})
        gen = LLMCandidateGenerator(chain=chain)
        out = gen("make a square", None)
        self.assertEqual(out, "G90\nM30")
        self.assertEqual(gen.last_usage.total, 14)

    def test_generator_handles_missing_usage(self):
        gen = LLMCandidateGenerator(chain=_FakeChain("G90\nM30"))
        gen("x", None)
        self.assertEqual(gen.last_usage.total, 0)

    def test_generate_prompt_includes_task_and_constraints(self):
        req = LoopRequest(prompt="mill a square")
        p = build_generate_prompt(req, ("allowed_tools: T1",))
        self.assertIn("mill a square", p)
        self.assertIn("allowed_tools: T1", p)

    def test_repair_prompt_includes_findings_and_previous(self):
        req = LoopRequest(prompt="mill a square")
        findings = [Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", line_number=2)]
        p = build_repair_prompt(req, "T99 M06", findings, ("allowed_tools: T1",))
        self.assertIn("unsupported_tool", p)
        self.assertIn("T99 M06", p)
        self.assertIn("allowed_tools: T1", p)

    def test_improve_prompt_states_objective_and_value(self):
        req = LoopRequest(prompt="mill a square", objective=Objective("path_length"))
        p = build_improve_prompt(req, "G1 X1 Y1", Objective("path_length"), 12.5, ())
        self.assertIn("path_length", p)
        self.assertIn("12.5", p)
        self.assertIn("G1 X1 Y1", p)


if __name__ == "__main__":
    unittest.main()
