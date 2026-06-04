import subprocess
import unittest

from gllm.utils.codex_model import CodexCliLanguageModel


class CodexCliLanguageModelTests(unittest.TestCase):
    def test_invoke_sends_prompt_over_stdin_and_returns_stdout(self):
        calls = []

        def fake_runner(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="G00 X0 Y0\n", stderr="")

        model = CodexCliLanguageModel(runner=fake_runner, timeout_seconds=5)

        result = model.invoke("Generate safe G-code")

        self.assertEqual(result, "G00 X0 Y0")
        args, kwargs = calls[0]
        self.assertEqual(args[-1], "-")
        self.assertIn("--ephemeral", args)
        self.assertIn("--sandbox", args)
        self.assertEqual(kwargs["input"], "Generate safe G-code")
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    def test_invoke_formats_dict_inputs_like_langchain_chains(self):
        def fake_runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=kwargs["input"], stderr="")

        model = CodexCliLanguageModel(runner=fake_runner)

        result = model.invoke({"input": "Cut a circle"})

        self.assertEqual(result, "Cut a circle")

    def test_invoke_raises_sanitized_error_on_codex_failure(self):
        def fake_runner(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr="not logged in: token abc123",
            )

        model = CodexCliLanguageModel(runner=fake_runner)

        with self.assertRaisesRegex(RuntimeError, "Codex CLI model failed"):
            model.invoke("hello")


if __name__ == "__main__":
    unittest.main()
