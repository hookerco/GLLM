import unittest

from gllm.loop.controller import LoopController
from gllm.loop.types import LoopRequest, TokenUsage
from gllm.loop.findings import Finding, Severity


def _err(code="unsupported_tool"):
    return Finding("static", code, Severity.ERROR, "bad", blocking=True)


def _verdict(status, *, passed=None, errors=None, findings=None):
    """A verdict dict shaped like VericutVerdict.to_dict()."""
    if passed is None:
        passed = status == "vericut_accepted"
    if findings is None:
        findings = (
            [{"severity": "error", "code": "collision",
              "message": "Component X exceeded limit", "line_number": 4}]
            if status == "vericut_rejected"
            else []
        )
    if errors is None:
        errors = len(findings)
    return {
        "status": status,
        "passed": passed,
        "vericut": {"error_count": errors, "warning_count": 0},
        "repair_context": findings,
    }


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


class _ScriptedGate:
    """Returns a pre-scripted verdict dict per call; repeats the last one."""

    def __init__(self, verdicts):
        self._verdicts = verdicts
        self.calls = 0

    def __call__(self, gcode, ctx, request):
        v = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return v


def _ctrl(gen, val, gate):
    return LoopController(generator=gen, blocking_validators=[val], final_gate=gate)


def _req(**kw):
    base = dict(prompt="p", run_vericut=True, max_attempts=1)
    base.update(kw)
    return LoopRequest(**base)


class VericutRepairTests(unittest.TestCase):
    def test_rejected_then_repairs_to_acceptance(self):
        gate = _ScriptedGate([_verdict("vericut_rejected"), _verdict("vericut_accepted")])
        gen = _Gen(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"])
        result = _ctrl(gen, _Val([[]]), gate).run(_req(vericut_max_rounds=2))
        self.assertEqual(result.status, "accepted_vericut")
        self.assertEqual(gate.calls, 2)  # initial gate + 1 repair
        self.assertEqual(len(result.history), 2)  # original + repaired
        self.assertEqual(result.best.gcode, "G1 X2 Y2 F50\nM30")
        self.assertEqual(result.vericut["status"], "vericut_accepted")

    def test_exhausts_cap_and_stays_rejected(self):
        gate = _ScriptedGate([_verdict("vericut_rejected")])  # always rejects
        result = _ctrl(_Gen(["G1 X1 Y1 F50\nM30"]), _Val([[]]), gate).run(_req(vericut_max_rounds=2))
        self.assertEqual(result.status, "rejected_vericut")
        self.assertEqual(gate.calls, 3)  # initial + 2 repair rounds

    def test_max_rounds_zero_preserves_single_gate(self):
        gate = _ScriptedGate([_verdict("vericut_rejected")])
        result = _ctrl(_Gen(["G1 X1 Y1 F50\nM30"]), _Val([[]]), gate).run(_req(vericut_max_rounds=0))
        self.assertEqual(result.status, "rejected_vericut")
        self.assertEqual(gate.calls, 1)
        self.assertEqual(len(result.history), 1)

    def test_best_pair_keeps_fewest_errors(self):
        gate = _ScriptedGate([_verdict("vericut_rejected", errors=3), _verdict("vericut_rejected", errors=1)])
        gen = _Gen(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"])
        result = _ctrl(gen, _Val([[]]), gate).run(_req(vericut_max_rounds=1))
        self.assertEqual(result.status, "rejected_vericut")
        self.assertEqual(result.vericut["vericut"]["error_count"], 1)
        self.assertEqual(result.best.gcode, "G1 X2 Y2 F50\nM30")

    def test_static_regression_during_repair_skips_simulation(self):
        gate = _ScriptedGate([_verdict("vericut_rejected")])
        gen = _Gen(["G1 X1 Y1 F50\nM30", "T99 M06"])  # repaired candidate regresses
        val = _Val([[], [_err()]])  # attempt1 clean, repaired attempt has a blocking finding
        result = _ctrl(gen, val, gate).run(_req(vericut_max_rounds=1))
        self.assertEqual(gate.calls, 1)  # repaired candidate never simulated
        self.assertEqual(result.status, "rejected_vericut")
        self.assertEqual(len(result.history), 2)

    def test_token_budget_halts_mid_repair(self):
        gate = _ScriptedGate([_verdict("vericut_rejected")])  # always rejects
        # static generate spends 2; budget 3 admits exactly one repair generation.
        result = _ctrl(_Gen(["G1 X1 Y1 F50\nM30"]), _Val([[]]), gate).run(
            _req(vericut_max_rounds=5, token_budget=3)
        )
        self.assertEqual(gate.calls, 2)  # initial + 1 repair, then budget stops it
        self.assertEqual(result.status, "rejected_vericut")

    def test_unverified_does_not_trigger_repair(self):
        gate = _ScriptedGate([_verdict("vericut_unverified", passed=False)])
        result = _ctrl(_Gen(["G1 X1 Y1 F50\nM30"]), _Val([[]]), gate).run(_req(vericut_max_rounds=2))
        self.assertEqual(gate.calls, 1)
        self.assertEqual(result.status, "blocked_vericut_unverified")
        self.assertEqual(len(result.history), 1)

    def test_repair_emits_events_per_round(self):
        gate = _ScriptedGate([_verdict("vericut_rejected"), _verdict("vericut_accepted")])
        gen = _Gen(["G1 X1 Y1 F50\nM30", "G1 X2 Y2 F50\nM30"])
        types = [e.type for e in _ctrl(gen, _Val([[]]), gate).stream(_req(vericut_max_rounds=2))]
        self.assertEqual(types.count("vericut_started"), 2)
        self.assertEqual(types.count("vericut_verdict"), 2)
        # a repair prompt is emitted between the two simulations
        first = types.index("vericut_verdict")
        self.assertIn("repair_prompt", types[first:])
        self.assertEqual(types[-1], "done")


if __name__ == "__main__":
    unittest.main()
