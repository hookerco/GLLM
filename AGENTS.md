# AGENTS.md

## Purpose

This repo is a Python/Streamlit project for generating, validating, and visualizing CNC G-code with LLM-backed pipelines. Treat it as an application plus research artifact: keep changes scoped, preserve the checked-in data/model evidence unless the user explicitly asks to prune it, and make authentication safer before expanding model behavior.

## Read First

- `README.md` for the published setup and app entrypoint.
- `pyproject.toml` for the Python and dependency contract.
- `gllm/code_generator_streamlit_reasoning_langchain_langgraph.py` for the Streamlit UI entrypoint.
- `gllm/utils/model_utils.py`, `gllm/utils/rag_utils.py`, and `helpers/chatbot.py` for language-model connection code.

## Environment

- The project requires Python 3.11 via Poetry: `python = "~3.11"`.
- On Windows, prefer selecting the installed 3.11 interpreter explicitly:

```powershell
poetry env use (py -3.11 -c "import sys; print(sys.executable)")
poetry install
```

- The app command is:

```powershell
poetry run streamlit run gllm/code_generator_streamlit_reasoning_langchain_langgraph.py
```

- Do not assume the current shell `python` is usable; this workspace has other Python versions available.
- If dependency installation requires network access, ask for approval before escalating.

## Secrets And Auth

- Never commit `.streamlit/secrets.toml`, API keys, OAuth tokens, refresh tokens, or copied credential material.
- `.streamlit/secrets.toml.example` is the only Streamlit secrets file that should be tracked.
- Existing code supports `openai_token` and `huggingface_token` style Streamlit secrets as a fallback, but environment variables should win when present.
- Codex OAuth is supported through the `Codex OAuth` model option, which shells out to `codex exec` and lets Codex consume its own cached login, `CODEX_ACCESS_TOKEN`, or `CODEX_API_KEY`. Do not hardcode Codex credentials, read `~/.codex/auth.json`, or serialize session tokens.
- For ordinary OpenAI API calls such as `ChatOpenAI` and embeddings, keep using Platform API-key auth (`OPENAI_API_KEY` or `openai_token`). Codex access tokens are for Codex local workflows, not a general OpenAI API-key substitute.

## Editing Boundaries

- Keep UI work in the Streamlit entrypoint and nearby `gllm/utils/*` modules unless the user asks for a larger redesign.
- Keep training and dataset changes separate from app/auth changes.
- Do not modify `finetuned_model/`, `faiss_index/`, `data/pdfs/`, `data/txt/`, or notebooks unless the task specifically targets those artifacts.
- Avoid broad refactors in `helpers/` unless they are needed to unblock a requested workflow.
- Do not add generated caches, virtualenvs, downloaded model files, or local secrets to git.

## Current Codex Friction Points

- Poetry may select a non-3.11 interpreter unless explicitly configured.
- Model authentication is now resolved lazily through `gllm/utils/auth_utils.py`; do not reintroduce import-time secret reads.
- There is no dedicated test suite in the repo yet, so verification should be explicit about what was and was not proven.

## Verification

For doc-only changes:

```powershell
git diff --check
```

For packaging/config checks:

```powershell
poetry check
```

After dependencies are installed, use a lightweight syntax/import-adjacent check before launching the app:

```powershell
poetry run python -m compileall gllm helpers
```

For Streamlit runtime checks, launch the app with the documented command and inspect the browser/UI. Do not claim model generation works unless a real model path and credentials were exercised.

## Communication Rules For Agents

- State exact commands run and whether they passed, warned, or were blocked.
- Call out missing credentials, missing Python 3.11 env setup, and network-dependent dependency installs as separate blockers.
- When answering model-auth questions, distinguish confirmed repo behavior from planned Codex OAuth behavior.
- Prefer small, reviewable changes that make the next Codex turn easier.
