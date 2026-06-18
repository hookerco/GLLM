import unittest
from gllm.loop.controller import LoopController, pick_best, derive_status
from gllm.loop.types import LoopRequest, Mode, Objective, Attempt
from gllm.loop.findings import Finding, Severity


def _err(code="x"):
    return Finding("static", code, Severity.ERROR, "bad", blocking=True)


class _ScriptedValidator:
    """Returns a pre-scripted findings list per attempt index (0-based)."""
    name = "scripted"
    tier = "static"

    def __init__(self, per_attempt):
        self._per_attempt = per_attempt
        self._i = -1

    def validate(self, gcode, ctx):
        self._i += 1
        idx = min(self._i, len(self._per_attempt) - 1)
        return list(self._per_attempt[idx])


class _ScriptedGenerator:
    def __init__(self, outputs):
        self._outputs = outputs
        self._i = -1
        from gllm.loop.types import TokenUsage
        self.last_usage = TokenUsage(1, 1)

    def __call__(self, prompt, ctx=None):
        self._i += 1
        return self._outputs[min(self._i, len(self._outputs) - 1)]


class ControllerLoopTests(unittest.TestCase):
    def _controller(self, gen, validators):
        return LoopController(generator=gen, blocking_validators=validators, final_gate=None)

    def test_converges_and_stops_early(self):
        gen = _ScriptedGenerator(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"])
        val = _ScriptedValidator([[_err()], []])  # attempt2 clean
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=5))
        self.assertEqual(result.status, "passed")
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.best.gcode, "G1 X2 Y2 F50\nM30")

    def test_caps_at_max_attempts_and_returns_best_effort(self):
        gen = _ScriptedGenerator(["a", "b", "c"])
        val = _ScriptedValidator([[_err(), _err("y")], [_err()], [_err()]])
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=2))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.status, "exhausted_best_effort")
        # best = attempt 2 (1 error) over attempt 1 (2 errors)
        self.assertEqual(result.best.number, 2)

    def test_patience_stops_when_no_improvement(self):
        gen = _ScriptedGenerator(["a", "b", "c", "d", "e"])
        val = _ScriptedValidator([[_err()]])  # every attempt: same single error
        result = self._controller(gen, [val]).run(LoopRequest(prompt="p", max_attempts=10, patience=2))
        # attempt1 sets best; attempts 2 and 3 don't improve -> stop after 3
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.status, "exhausted_best_effort")

    def test_repair_mode_uses_seed_as_attempt_one(self):
        gen = _ScriptedGenerator(["regenerated"])  # only used if a 2nd attempt happens
        val = _ScriptedValidator([[]])  # seed is already clean
        req = LoopRequest(prompt="p", mode=Mode.REPAIR, seed_gcode="G1 X1 Y1 F50\nM30", max_attempts=3)
        result = self._controller(gen, [val]).run(req)
        self.assertEqual(result.history[0].gcode, "G1 X1 Y1 F50\nM30")
        self.assertEqual(result.status, "passed")

    def test_improve_keeps_best_objective_among_green(self):
        # Both attempts green; improve picks the lower path_length one.
        gen = _ScriptedGenerator(["G0 X0 Y0\nG1 X10 Y0\nM30"])  # 2nd candidate shorter
        val = _ScriptedValidator([[]])
        req = LoopRequest(
            prompt="p", mode=Mode.IMPROVE, seed_gcode="G0 X0 Y0\nG1 X100 Y0\nM30",
            objective=Objective("path_length"), max_attempts=2, patience=5,
        )
        result = self._controller(gen, [val]).run(req)
        self.assertIn(result.status, ("improved", "not_improved"))
        self.assertEqual(result.best.gcode, "G0 X0 Y0\nG1 X10 Y0\nM30")
        self.assertEqual(result.status, "improved")


class PickBestStatusTests(unittest.TestCase):
    def test_pick_best_prefers_fewer_errors(self):
        a1 = Attempt(1, "a", (_err(), _err("y")), score=2000.0)
        a2 = Attempt(2, "b", (_err(),), score=1000.0)
        self.assertIs(pick_best(a1, a2, Mode.GENERATE, None), a2)
