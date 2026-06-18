import unittest
from gllm.loop import LoopController, LoopRequest, Mode, Objective, result_to_dict
from gllm.loop.findings import Finding, Severity


def _err():
    return Finding("static", "unsupported_tool", Severity.ERROR, "T99 not allowed", blocking=True)


class _Gen:
    def __init__(self, outs):
        from gllm.loop.types import TokenUsage
        self._outs, self._i, self.last_usage = outs, -1, TokenUsage(2, 2)

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


class EngineEndToEndTests(unittest.TestCase):
    def test_generate_repairs_then_passes(self):
        ctrl = LoopController(generator=_Gen(["T99 M06", "T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        result = ctrl.run(LoopRequest(prompt="mill a square", max_attempts=4))
        self.assertEqual(result.status, "passed")
        d = result_to_dict(result)
        self.assertEqual(d["best_attempt_index"], 2)
        self.assertEqual(d["mode"], "generate")

    def test_repair_seed_then_passes(self):
        ctrl = LoopController(generator=_Gen(["T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        req = LoopRequest(prompt="fix it", mode=Mode.REPAIR, seed_gcode="T99 M06", max_attempts=4)
        result = ctrl.run(req)
        self.assertEqual(result.history[0].gcode, "T99 M06")
        self.assertEqual(result.status, "passed")

    def test_improve_returns_seed_when_nothing_beats_it(self):
        # Generated candidate is LONGER -> seed stays best -> not_improved.
        ctrl = LoopController(generator=_Gen(["G1 X999 Y0\nM30"]), blocking_validators=[_Val([[]])])
        req = LoopRequest(prompt="optimize", mode=Mode.IMPROVE, seed_gcode="G1 X1 Y0\nM30",
                          objective=Objective("path_length"), max_attempts=2, patience=5)
        result = ctrl.run(req)
        self.assertEqual(result.best.gcode, "G1 X1 Y0\nM30")
        self.assertEqual(result.status, "not_improved")

    def test_streaming_emits_expected_event_sequence(self):
        ctrl = LoopController(generator=_Gen(["T99 M06", "T1 M06"]), blocking_validators=[_Val([[_err()], []])])
        types = [e.type for e in ctrl.stream(LoopRequest(prompt="p", max_attempts=4))]
        self.assertEqual(types[0], "attempt_started")
        self.assertIn("repair_prompt", types)
        self.assertEqual(types[-1], "done")
