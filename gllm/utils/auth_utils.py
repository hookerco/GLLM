"""
Authentication helpers for model providers.

The helpers keep credential lookup lazy so importing model modules does not
require local secrets to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping, Optional


@dataclass(frozen=True)
class AuthStatus:
    available: bool
    source: Optional[str] = None


def _first_nonempty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _mapping_get(mapping: Optional[Mapping[str, str]], key: str) -> Optional[str]:
    if mapping is None:
        return None
    try:
        value = mapping.get(key)
    except AttributeError:
        return None
    return value or None


def load_streamlit_secrets() -> Mapping[str, str]:
    try:
        import streamlit as st

        return st.secrets
    except Exception:
        return {}


def load_toml_secrets(path: Path | str = ".streamlit/secrets.toml") -> Mapping[str, str]:
    secrets_path = Path(path)
    if not secrets_path.exists():
        return {}

    try:
        import toml

        loaded = toml.load(secrets_path)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _default_secrets() -> Mapping[str, str]:
    streamlit_secrets = load_streamlit_secrets()
    if streamlit_secrets:
        return streamlit_secrets
    return load_toml_secrets()


def resolve_openai_api_key(
    *,
    env: Optional[Mapping[str, str]] = None,
    secrets: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    env = os.environ if env is None else env
    secrets = _default_secrets() if secrets is None else secrets
    return _first_nonempty(
        _mapping_get(env, "OPENAI_API_KEY"),
        _mapping_get(env, "OPENAI_TOKEN"),
        _mapping_get(secrets, "openai_token"),
    )


def resolve_huggingface_token(
    *,
    env: Optional[Mapping[str, str]] = None,
    secrets: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    env = os.environ if env is None else env
    secrets = _default_secrets() if secrets is None else secrets
    return _first_nonempty(
        _mapping_get(env, "HUGGINGFACEHUB_API_TOKEN"),
        _mapping_get(env, "HF_TOKEN"),
        _mapping_get(secrets, "huggingface_token"),
    )


def describe_codex_auth(*, env: Optional[Mapping[str, str]] = None) -> AuthStatus:
    env = os.environ if env is None else env
    if _mapping_get(env, "CODEX_ACCESS_TOKEN"):
        return AuthStatus(available=True, source="CODEX_ACCESS_TOKEN")
    if _mapping_get(env, "CODEX_API_KEY"):
        return AuthStatus(available=True, source="CODEX_API_KEY")

    codex_home = Path(_mapping_get(env, "CODEX_HOME") or Path.home() / ".codex")
    auth_file = codex_home / "auth.json"
    if auth_file.exists():
        return AuthStatus(available=True, source=str(auth_file))

    return AuthStatus(available=False)
