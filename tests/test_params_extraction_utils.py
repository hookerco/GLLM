import json
import unittest

from gllm.utils.params_extraction_utils import (
    extract_parameters_with_langchain,
    parse_extracted_parameters,
)


class BrokenJsonChain:
    def invoke(self, _payload):
        raise json.JSONDecodeError("Expecting value", "data:\n", 0)


class ParamsExtractionUtilsTests(unittest.TestCase):
    def test_non_json_model_response_is_reported_as_model_provider_error(self):
        with self.assertRaisesRegex(RuntimeError, "non-JSON response"):
            extract_parameters_with_langchain(
                BrokenJsonChain(),
                "Mill a square pocket in aluminum.",
            )

    def test_donut_tool_path_coordinates_are_parsed_from_labeled_points(self):
        extracted_text = (
            "Material: 4150ht\n"
            "Operation Type: milling (pocketing/profiling)\n"
            "Desired Shape: donut (annulus)\n"
            'Workpiece Dimensions: 12" x 12"\n'
            "Starting Point: 0, 0, 0.1\n"
            "Home Position: 0, 0, 0\n"
            'Cutting Tool Path: 1. Outer Circle: center (0,0), radius 3" '
            "(x3.0, y0.0 to x0.0, y3.0 to x-3.0, y0.0 to x0.0, y-3.0 to x3.0, y0.0) "
            '2. Inner Circle: center (0,0), radius 1" '
            "(x1.0, y0.0 to x0.0, y1.0 to x-1.0, y0.0 to x0.0, y-1.0 to x1.0, y0.0)\n"
            "Return Tool to Home After Execution: yes\n"
            'Depth of Cut: 0.25" (inferred)\n'
            "Feed Rate: 20 ipm (inferred)\n"
            "Spindle Speed: 1200 rpm (inferred)\n"
            'Radius: 3" (outer), 1" (inner - inferred for donut hole)\n'
            "Number of Shapes: 1\n"
        )

        parsed = parse_extracted_parameters(extracted_text)

        self.assertEqual(parsed["starting_point"], [0.0, 0.0, 0.1])
        self.assertEqual(parsed["cut_depth"], [0.25])
        self.assertEqual(
            parsed["tool_path"],
            [
                (3.0, 0.0, 0.0),
                (0.0, 3.0, 0.0),
                (-3.0, 0.0, 0.0),
                (0.0, -3.0, 0.0),
                (3.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
