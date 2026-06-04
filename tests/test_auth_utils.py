import unittest

from gllm.utils.auth_utils import (
    AuthStatus,
    describe_codex_auth,
    resolve_huggingface_token,
    resolve_openai_api_key,
    resolve_openrouter_api_key,
    resolve_streamlit_secret,
)


class AuthUtilsTests(unittest.TestCase):
    def test_openai_api_key_prefers_environment(self):
        key = resolve_openai_api_key(
            env={"OPENAI_API_KEY": "from-env"},
            secrets={"openai_token": "from-secrets"},
        )

        self.assertEqual(key, "from-env")

    def test_openai_api_key_falls_back_to_streamlit_secret_name(self):
        key = resolve_openai_api_key(env={}, secrets={"openai_token": "from-secrets"})

        self.assertEqual(key, "from-secrets")

    def test_missing_openai_api_key_returns_none(self):
        key = resolve_openai_api_key(env={}, secrets={})

        self.assertIsNone(key)

    def test_huggingface_token_uses_existing_environment_first(self):
        token = resolve_huggingface_token(
            env={"HUGGINGFACEHUB_API_TOKEN": "from-env"},
            secrets={"huggingface_token": "from-secrets"},
        )

        self.assertEqual(token, "from-env")

    def test_codex_access_token_status_does_not_expose_secret_value(self):
        status = describe_codex_auth(env={"CODEX_ACCESS_TOKEN": "secret-value"})

        self.assertEqual(status, AuthStatus(available=True, source="CODEX_ACCESS_TOKEN"))
        self.assertNotIn("secret-value", repr(status))

    def test_openrouter_api_key_prefers_environment(self):
        key = resolve_openrouter_api_key(
            env={"OPENROUTER_API_KEY": "from-env"},
            secrets={"openrouter_token": "from-secrets"},
        )

        self.assertEqual(key, "from-env")

    def test_streamlit_secret_reads_named_value(self):
        value = resolve_streamlit_secret("openrouter_model", secrets={"openrouter_model": "openai/gpt-4o"})

        self.assertEqual(value, "openai/gpt-4o")


if __name__ == "__main__":
    unittest.main()
