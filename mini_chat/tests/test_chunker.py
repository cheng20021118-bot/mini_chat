import unittest

from rag.chunker import split_text


class TestChunker(unittest.TestCase):
    def test_short_text(self):
        text = "hello"
        out = split_text(text, chunk_size=100, overlap=10)
        self.assertEqual(out, ["hello"])

    def test_overlap(self):
        text = "a" * 500
        out = split_text(text, chunk_size=200, overlap=50)
        self.assertGreaterEqual(len(out), 2)
        # ensure chunks are not empty
        self.assertTrue(all(len(c) > 0 for c in out))


if __name__ == "__main__":
    unittest.main()
