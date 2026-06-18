from __future__ import annotations

import argparse
import json

from gllm.proof.corpus import load_prompt_corpus, run_prompt_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a prompt-to-verdict corpus.")
    parser.add_argument("--corpus", required=True, help="Path to a prompt corpus JSON file.")
    parser.add_argument("--registry", required=True, help="Path to a Vericut setup registry JSON.")
    parser.add_argument(
        "--output-root",
        default=".proof-runs/corpus",
        help="Directory where corpus proof run folders are written.",
    )
    args = parser.parse_args()

    corpus = load_prompt_corpus(args.corpus)
    result = run_prompt_corpus(
        corpus,
        registry_path=args.registry,
        output_root=args.output_root,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
