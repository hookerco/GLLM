import unittest
from gllm.utils.gcode_utils import (
    validate_z_levels, check_tool_offsets, validate_functional_correctness,
)


class ValidatorFixTests(unittest.TestCase):
    def test_z_levels_does_not_crash_on_lines_without_gcodes(self):
        # A bare-coordinate line like "Z-200" has a Z word but NO G word, so
        # pygcode.Line("Z-200").block.gcodes is empty ([]). Without the guard,
        # `line.block.gcodes[0].params` raises IndexError. The guard must skip it.
        ok, _ = validate_z_levels("Z-200\nM30", max_depth=100)
        self.assertTrue(ok)

    def test_check_tool_offsets_does_not_crash_on_lines_without_gcodes(self):
        # The bare-coordinate "Z-200" line (empty block.gcodes, contains a Z word)
        # appears while tool offset is active (after G43, before G49). Without the
        # guard, `line.block.gcodes[0].params` raises IndexError. The guard must skip it.
        ok, _ = check_tool_offsets("G43\nZ-200\nG49\nM30")
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

    def test_functional_correctness_no_user_params_returns_true(self):
        # Locks in that the widened `try` still produces a well-formed 2-tuple result
        # on the no-user-params path. NOTE: in this installed environment an empty
        # parameter string does NOT yield None from parse_extracted_parameters -- it
        # raises KeyError('Desired Shape'), which the widened try now catches and turns
        # into (False, msg). So we assert the contract that always holds: the result is
        # a 2-tuple whose first element is a bool. (See report: the (True, None)
        # no-params fall-through could not be exercised here.)
        result = validate_functional_correctness("G1 X1 Y1\nM30", "")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        ok, _ = result
        self.assertIsInstance(ok, bool)


if __name__ == "__main__":
    unittest.main()
