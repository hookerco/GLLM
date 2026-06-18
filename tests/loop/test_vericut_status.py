import unittest

from gllm.loop.controller import LoopController, derive_status
from gllm.loop.types import Attempt, LoopRequest, Mode, TokenUsage
from gllm.loop.packet import operator_action


def _clean_attempt(n=1, gcode="G1 X1 Y1\nM30"):
    return Attempt(n, gcode, (), 0.0)


class _Gen:
    def __init__(self, outs):
        self._outs, self._i, self.last_usage = outs, -1, TokenUsage(1, 1)

    def __call__(self, prompt, ctx=None):
        self._i += 1
        return self._outs[min(self._i, len(self._outs) - 1)]


class _Val:
    name, tier = "scripted", "static"

    def __init__(self, per):
        self._per, self._i = per, -1

    def validate(self, gcode, ctx):
        self._i += 1
        return list(self._per[min(self._i, len(self._per) - 1)])


class DeriveStatusTests(unittest.TestCase):
    def test_unavailable_status(self):
        self.assertEqual(
            derive_status(_clean_attempt(), Mode.GENERATE, {"status": "vericut_unavailable"}),
            "passed_vericut_unavailable",
        )

    def test_passed_when_no_vericut(self):
        self.assertEqual(derive_status(_clean_attempt(), Mode.GENERATE, None), "passed")

    def test_accepted_when_passed(self):
        self.assertEqual(
            derive_status(_clean_attempt(), Mode.GENERATE, {"passed": True, "status": "vericut_accepted"}),
            "accepted_vericut",
        )

    def test_rejected_when_failed(self):
        self.assertEqual(
            derive_status(_clean_attempt(), Mode.GENERATE, {"passed": False, "status": "vericut_rejected"}),
            "rejected_vericut",
        )


class ControllerGateTests(unittest.TestCase):
    def _ctrl(self, final_gate):
        return LoopController(
            generator=_Gen(["G1 X1 Y1 F50\nM30"]),
            blocking_validators=[_Val([[]])],  # clean -> green best
            final_gate=final_gate,
        )

    def test_gate_unavailable_yields_unavailable_status(self):
        result = self._ctrl(lambda gcode, ctx, req: None).run(
            LoopRequest(prompt="p", run_vericut=True, max_attempts=1)
        )
        self.assertEqual(result.status, "passed_vericut_unavailable")
        self.assertEqual(result.vericut, {"status": "vericut_unavailable"})

    def test_gate_passed_yields_accepted(self):
        result = self._ctrl(lambda gcode, ctx, req: {"passed": True, "status": "vericut_accepted"}).run(
            LoopRequest(prompt="p", run_vericut=True, max_attempts=1)
        )
        self.assertEqual(result.status, "accepted_vericut")

    def test_no_vericut_requested_is_plain_passed(self):
        # gate would return None, but run_vericut is False so the gate is never called
        result = self._ctrl(lambda gcode, ctx, req: None).run(
            LoopRequest(prompt="p", run_vericut=False, max_attempts=1)
        )
        self.assertEqual(result.status, "passed")
        self.assertIsNone(result.vericut)


class OperatorActionTests(unittest.TestCase):
    def test_unavailable_maps_to_rerun_vericut(self):
        self.assertEqual(operator_action("passed_vericut_unavailable"), "rerun_vericut")


if __name__ == "__main__":
    unittest.main()
