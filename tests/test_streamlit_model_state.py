import subprocess
import sys
import unittest
from unittest.mock import patch

from gllm.code_generator_streamlit_reasoning_langchain_langgraph import (
    get_or_setup_model,
    select_proof_candidate_gcode,
)


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

    def test_manual_proof_candidate_wins_over_generated_gcode(self):
        self.assertEqual(
            select_proof_candidate_gcode(
                " G20\nG90\nM30\n",
                "G20\nG90\nT1 M06\nM30\n",
            ),
            "G20\nG90\nM30",
        )

    def test_generated_gcode_is_proof_candidate_fallback(self):
        self.assertEqual(
            select_proof_candidate_gcode("", " G20\nG90\nT1 M06\nM30\n"),
            "G20\nG90\nT1 M06\nM30",
        )

    def test_blank_proof_candidate_resolves_empty(self):
        self.assertEqual(select_proof_candidate_gcode("  ", None), "")

    def test_existing_gcode_proof_helper_uses_current_generated_candidate(self):
        from gllm.code_generator_streamlit_reasoning_langchain_langgraph import (
            run_existing_gcode_proof,
        )

        expected_packet = object()
        with patch(
            "gllm.code_generator_streamlit_reasoning_langchain_langgraph.run_proof_scenario",
            return_value=expected_packet,
        ) as proof_runner:
            packet = run_existing_gcode_proof(
                prompt="Mill a square.",
                gcode="G90\nT1 M06\nG0 Z1.0\nM30\n",
                registry_path="config/vericut_setups.example.json",
                setup_id="sample_haas",
                output_root=".proof-runs",
                run_vericut=True,
                model_name="OpenRouter",
                timeout_seconds=120,
                scenario_id="ui-proof",
            )

        self.assertIs(packet, expected_packet)
        request = proof_runner.call_args.args[0]
        candidate_generator = proof_runner.call_args.kwargs["candidate_generator"]
        self.assertEqual(request.prompt, "Mill a square.")
        self.assertEqual(request.setup_id, "sample_haas")
        self.assertEqual(request.scenario_id, "ui-proof")
        self.assertTrue(request.run_vericut)
        self.assertEqual(request.timeout_seconds, 120)
        self.assertEqual(
            candidate_generator("ignored repair prompt", object()),
            "G90\nT1 M06\nG0 Z1.0\nM30\n",
        )

    def test_proof_verdict_card_prioritizes_operator_action_and_evidence_path(self):
        from gllm.code_generator_streamlit_reasoning_langchain_langgraph import (
            proof_verdict_card,
        )

        card = proof_verdict_card(
            {
                "status": "accepted_static_only",
                "operator_action": "rerun_vericut",
                "final_attempt": 1,
                "packet_file": ".proof-runs/sample/evidence_packet.json",
                "attempts": [{"status": "accepted_static_only"}],
            }
        )

        self.assertEqual(card["severity"], "warning")
        self.assertEqual(card["headline"], "accepted_static_only")
        self.assertEqual(card["operator_action"], "rerun_vericut")
        self.assertEqual(card["attempts"], 1)
        self.assertEqual(card["evidence_packet"], ".proof-runs/sample/evidence_packet.json")


if __name__ == "__main__":
    unittest.main()
