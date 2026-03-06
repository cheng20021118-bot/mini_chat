import unittest

from rag.query import normalize_query


class TestNormalizeQuery(unittest.TestCase):
    def test_remove_stop_patterns(self):
        q = "请问 Transformer 是什么？"
        out = normalize_query(q)
        self.assertIn("Transformer", out)
        self.assertNotIn("请问", out)
        self.assertNotIn("是什么", out)

    def test_fallback(self):
        q = "？"
        out = normalize_query(q)
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
