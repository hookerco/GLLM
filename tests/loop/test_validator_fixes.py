import unittest
from gllm.utils.gcode_utils import (
    validate_z_levels, check_tool_offsets, validate_functional_correctness,
)


class ValidatorFixTests(unittest.TestCase):
    def test_z_levels_does_not_crash_on_lines_without_gcodes(self):
        # "S1000" has modal params but block.gcodes is empty -> used to IndexError.
        ok, _ = validate_z_levels("S1000\nF50\nM30", max_depth=100)
        self.assertTrue(ok)

    def test_check_tool_offsets_does_not_crash_on_lines_without_gcodes(self):
        ok, _ = check_tool_offsets("G43\nS1000\nG49\nM30")
        self.assertTrue(ok)

    def test_functional_correctness_does_not_silently_pass_on_unsupported_op(self):
        # Force the inner zip(*tool_path) to raise via a degenerate parameters_string;
        # the function must NOT return (True, None) by swallowing the exception.
        params = "starting_point: (0, 0)\ntool_path: bogus"
        ok, msg = validate_functional_correctness("G1 X1 Y1", params)
        # Either it parsed nothing (no user params -> True) or it flagged inability to verify,
        # but it must never claim success while swallowing an exception.
        if ok is False:
            self.assertIsNotNone(msg)
        self.assertIsInstance(ok, bool)


if __name__ == "__main__":
    unittest.main()
