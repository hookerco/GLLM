"""Launch the GLLM loop service + UI.

    poetry run python -m gllm.service            # http://127.0.0.1:8080
    poetry run python -m gllm.service --port 9000
"""
from __future__ import annotations

import argparse

import uvicorn

from gllm.service.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="GLLM loop service + UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
