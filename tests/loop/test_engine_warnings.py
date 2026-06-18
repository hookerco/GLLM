import unittest
from gllm.loop import LoopController, LoopRequest
from gllm.loop.types import TokenUsage


class _Gen:
    """Returns a program that passes all BLOCKING lint checks but trips the
    non-blocking 'continuity' warning (consecutive X/Y positions differ)."""
    last_usage = TokenUsage(1, 1)

    def __call__(self, prompt, ctx=None):
        return "G1 X1 Y1 F50\nG1 X9 Y9 F50\nM30"


class EngineWarningsTests(unittest.TestCase):
    def test_passed_with_warnings_through_default_validators(self):
        # No blocking_validators passed -> controller uses the real defaults
        # (LintValidator + StaticPolicyValidator). No setup -> static is a no-op.
        ctrl = LoopController(generator=_Gen())
        result = ctrl.run(LoopRequest(prompt="mill a diagonal", max_attempts=1))
        self.assertEqual(result.status, "passed_with_warnings")
        warn_codes = {f.code for f in result.best.findings if not f.blocking}
        self.assertIn("continuity", warn_codes)
        # And there are no blocking findings.
        self.assertEqual(result.best.blocking_findings, ())
