import unittest
from unittest.mock import patch

from gllm.utils.model_utils import setup_model


class ModelUtilsTests(unittest.TestCase):
    def test_openrouter_defaults_to_free_router(self):
        with (
            patch("gllm.utils.model_utils.resolve_openrouter_api_key", return_value="key"),
            patch("gllm.utils.model_utils.resolve_streamlit_secret", return_value=None),
            patch("gllm.utils.model_utils.ChatOpenAI") as chat_openai,
        ):
            setup_model("OpenRouter")

        _, kwargs = chat_openai.call_args
        self.assertEqual(kwargs["model"], "openrouter/free")

    def test_openrouter_uses_configured_model_secret(self):
        with (
            patch("gllm.utils.model_utils.resolve_openrouter_api_key", return_value="key"),
            patch(
                "gllm.utils.model_utils.resolve_streamlit_secret",
                return_value="qwen/qwen3-coder:free",
            ),
            patch("gllm.utils.model_utils.ChatOpenAI") as chat_openai,
        ):
            setup_model("OpenRouter")

        _, kwargs = chat_openai.call_args
        self.assertEqual(kwargs["model"], "qwen/qwen3-coder:free")


if __name__ == "__main__":
    unittest.main()
