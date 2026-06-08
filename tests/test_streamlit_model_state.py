import subprocess
import sys
import unittest

from gllm.code_generator_streamlit_reasoning_langchain_langgraph import get_or_setup_model


class StreamlitModelStateTests(unittest.TestCase):
    def test_app_import_does_not_load_optional_rag_or_unused_plotly_express(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import gllm.code_generator_streamlit_reasoning_langchain_langgraph; "
                    "print('gllm.utils.rag_utils' in sys.modules); "
                    "print('plotly.express' in sys.modules)"
                ),
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(completed.stdout.splitlines(), ["False", "False"])

    def test_reuses_model_until_selection_changes(self):
        calls = []

        def setup_model(model_name, openrouter_model_name=None):
            calls.append((model_name, openrouter_model_name))
            return f"model:{model_name}"

        state = {}

        first_model = get_or_setup_model(state, "OpenRouter", setup_model)
        second_model = get_or_setup_model(state, "OpenRouter", setup_model)

        self.assertEqual(first_model, "model:OpenRouter")
        self.assertIs(first_model, second_model)
        self.assertEqual(calls, [("OpenRouter", None)])

        state["langchain_chain"] = object()
        changed_model = get_or_setup_model(state, "Codex OAuth", setup_model)

        self.assertEqual(changed_model, "model:Codex OAuth")
        self.assertEqual(calls, [("OpenRouter", None), ("Codex OAuth", None)])
        self.assertNotIn("langchain_chain", state)

    def test_openrouter_model_name_changes_cached_model(self):
        calls = []

        def setup_model(model_name, openrouter_model_name=None):
            calls.append((model_name, openrouter_model_name))
            return f"model:{model_name}:{openrouter_model_name}"

        state = {}

        first_model = get_or_setup_model(
            state,
            "OpenRouter",
            setup_model,
            openrouter_model_name="openrouter/free",
        )
        second_model = get_or_setup_model(
            state,
            "OpenRouter",
            setup_model,
            openrouter_model_name="openrouter/free",
        )

        self.assertIs(first_model, second_model)
        self.assertEqual(calls, [("OpenRouter", "openrouter/free")])

        state["langchain_chain"] = object()
        changed_model = get_or_setup_model(
            state,
            "OpenRouter",
            setup_model,
            openrouter_model_name="qwen/qwen3-coder:free",
        )

        self.assertEqual(changed_model, "model:OpenRouter:qwen/qwen3-coder:free")
        self.assertEqual(
            calls,
            [
                ("OpenRouter", "openrouter/free"),
                ("OpenRouter", "qwen/qwen3-coder:free"),
            ],
        )
        self.assertNotIn("langchain_chain", state)


if __name__ == "__main__":
    unittest.main()
