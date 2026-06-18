import unittest
from gllm.loop.constraints import setup_constraints
from gllm.vericut.registry import load_setup_registry


class _StubSetup:
    id = "stub"
    allowed_tools = frozenset({"T1", "T2"})
    required_modes = frozenset({"G90"})
    expected_units = "inch"
    feed_rate_min = 1.0
    feed_rate_max = 100.0
    spindle_speed_min = None
    spindle_speed_max = 900.0
    work_envelope = None
    safe_z_min = 0.25


class ConstraintsTests(unittest.TestCase):
    def test_includes_tools_modes_units_feed_spindle_safe_z(self):
        c = setup_constraints(_StubSetup())
        joined = "\n".join(c)
        self.assertIn("allowed_tools: T1, T2", joined)
        self.assertIn("required_modes: G90", joined)
        self.assertIn("expected_units: inch", joined)
        self.assertIn("feed_rate_range: [1, 100]", joined)
        self.assertIn("spindle_speed_range: [-inf, 900]", joined)
        self.assertIn("safe_z_min: 0.25", joined)
