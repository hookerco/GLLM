import unittest
import subprocess
import sys
from unittest.mock import patch

from gllm.utils.model_utils import (
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    OPENROUTER_DEFAULT_MODEL,
    get_openrouter_model_name,
    setup_model,
)


class ModelUtilsTests(unittest.TestCase):
    def test_import_does_not_load_local_model_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import gllm.utils.model_utils; "
                    "print('peft' in sys.modules); "
                    "print('transformers' in sys.modules)"
                ),
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(completed.stdout.splitlines(), ["False", "False"])

    def test_openrouter_is_default_operator_model(self):
        self.assertEqual(DEFAULT_MODEL, "OpenRouter")
        self.assertEqual(MODEL_OPTIONS[0], "OpenRouter")
        self.assertIn("Codex OAuth", MODEL_OPTIONS)

    def test_openrouter_model_name_defaults_to_free_router(self):
        with patch("gllm.utils.model_utils.resolve_streamlit_secret", return_value=None):
            model_name = get_openrouter_model_name()

        self.assertEqual(model_name, OPENROUTER_DEFAULT_MODEL)

    def test_openrouter_defaults_to_free_router(self):
        with (
            patch("gllm.utils.model_utils.resolve_openrouter_api_key", return_value="key"),
            patch("gllm.utils.model_utils.resolve_streamlit_secret", return_value=None),
            patch("gllm.utils.model_utils.ChatOpenAI") as chat_openai,
        ):
            setup_model("OpenRouter")

        _, kwargs = chat_openai.call_args
        self.assertEqual(kwargs["model"], OPENROUTER_DEFAULT_MODEL)

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
