import unittest

from rag.gate import should_reject


class TestGate(unittest.TestCase):
    def test_reject_empty(self):
        self.assertTrue(should_reject([]))

    def test_reject_low_score(self):
        results = [(0.1, "doc1"), (0.05, "doc2")]
        self.assertTrue(should_reject(results, abs_th=0.2))

    def test_accept_high_score(self):
        results = [(0.9, "doc1"), (0.2, "doc2")]
        self.assertFalse(should_reject(results, abs_th=0.2))


if __name__ == "__main__":
    unittest.main()
