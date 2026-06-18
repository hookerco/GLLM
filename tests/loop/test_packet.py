import unittest
from gllm.loop.packet import result_to_dict
from gllm.loop.types import LoopRequest, LoopResult, Attempt, Mode, Objective
from gllm.loop.findings import Finding, Severity


class PacketTests(unittest.TestCase):
    def _result(self):
        a1 = Attempt(1, "BAD", (Finding("static", "unsupported_tool", Severity.ERROR, "T99", line_number=2),), 1000.0)
        a2 = Attempt(2, "GOOD", (), 0.0)
        req = LoopRequest(prompt="mill", mode=Mode.GENERATE, setup_id="s1")
        return LoopResult(req, "sc-1", "passed", best=a2, history=(a1, a2))

    def test_top_level_shape(self):
        d = result_to_dict(self._result())
        self.assertEqual(d["scenario_id"], "sc-1")
        self.assertEqual(d["status"], "passed")
        self.assertEqual(d["mode"], "generate")
        self.assertEqual(d["best_attempt_index"], 2)
        self.assertEqual(d["operator_action"], "ready_to_review")
        self.assertEqual(len(d["attempts"]), 2)

    def test_attempt_carries_findings_and_score(self):
        d = result_to_dict(self._result())
        first = d["attempts"][0]
        self.assertEqual(first["attempt_number"], 1)
        self.assertEqual(first["score"], 1000.0)
        self.assertEqual(first["findings"][0]["code"], "unsupported_tool")
