import unittest

from starlette.testclient import TestClient

from gllm.service.app import create_app
from gllm.service.runs import RunManager


class UISmokeTests(unittest.TestCase):
    def _client(self):
        # Default RunManager: its real controller factory is never invoked here
        # (no run is started), so no LLM is touched.
        return TestClient(create_app(RunManager()))

    def test_index_served(self):
        r = self._client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("GLLM", r.text)
        self.assertIn("/static/app.js", r.text)

    def test_static_js_served(self):
        r = self._client().get("/static/app.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("EventSource", r.text)

    def test_static_css_served(self):
        r = self._client().get("/static/style.css")
        self.assertEqual(r.status_code, 200)
        self.assertIn("--accent", r.text)


if __name__ == "__main__":
    unittest.main()
