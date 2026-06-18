import unittest

from gllm.utils.gcode_utils import clean_gcode


class GCodeUtilsTests(unittest.TestCase):
    def test_clean_gcode_preserves_numbered_machine_blocks(self):
        raw_gcode = """
Here is the program:
%
G20
(DRILLING TOOL PATH)
N0020T1M6
N0030G90G0X5.Y4.
N0050X2.
N0410X-.1Z2.F10.M9
%
""".strip()

        cleaned = clean_gcode(raw_gcode)

        self.assertEqual(
            cleaned.splitlines(),
            [
                "G20",
                "N0020T1M6",
                "N0030G90G0X5.Y4.",
                "N0050X2.",
                "N0410X-.1Z2.F10.M9",
            ],
        )


if __name__ == "__main__":
    unittest.main()
