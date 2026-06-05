import json
import unittest

from gllm.utils.params_extraction_utils import extract_parameters_with_langchain


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


if __name__ == "__main__":
    unittest.main()
