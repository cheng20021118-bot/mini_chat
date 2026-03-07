import json
import os
import tempfile
import unittest

from memory.store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def test_add_list_clear(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "memory.json")
            ms = MemoryStore(path)
            ms.add_memory("用户喜欢吃面条")
            ms.add_memory("用户喜欢吃面条")  # dedup
            self.assertEqual(ms.list_memories(), ["用户喜欢吃面条"])

            # reload
            ms2 = MemoryStore(path)
            self.assertEqual(ms2.list_memories(), ["用户喜欢吃面条"])

            ms2.clear()
            self.assertEqual(ms2.list_memories(), [])

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data, [])


if __name__ == "__main__":
    unittest.main()
