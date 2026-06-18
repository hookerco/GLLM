import unittest
from gllm.loop.types import Mode, Objective, TokenUsage, Budget, LoopRequest


class TypesTests(unittest.TestCase):
    def test_token_usage_total(self):
        self.assertEqual(TokenUsage(3, 5).total, 8)

    def test_budget_tracks_and_exhausts(self):
        b = Budget(max_attempts=4, token_budget=10, patience=2)
        self.assertFalse(b.exhausted())
        b.add(TokenUsage(4, 4))
        self.assertFalse(b.exhausted())
        b.add(TokenUsage(2, 1))
        self.assertTrue(b.exhausted())

    def test_budget_without_limit_never_exhausts(self):
        b = Budget(token_budget=None)
        b.add(TokenUsage(1_000_000, 0))
        self.assertFalse(b.exhausted())

    def test_loop_request_defaults(self):
        req = LoopRequest(prompt="mill a square")
        self.assertEqual(req.mode, Mode.GENERATE)
        self.assertIsNone(req.objective)
        self.assertEqual(req.max_attempts, 4)
