import json
import os
import time
import tempfile
from typing import Any, Dict, List, Tuple, Set


def _norm_text(s: str) -> str:
    """统一空白、去首尾空格。"""
    return " ".join((s or "").strip().split())


def _ngrams(s: str, n: int = 2) -> Set[str]:
    """字符 n-gram（对中文/英文都能工作）。"""
    s = _norm_text(s).lower().replace(" ", "")
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class MemoryStore:
    def __init__(self, path: str = "memory.json", max_items: int = 200):
        self.path = path
        self.max_items = max_items

        dir_ = os.path.dirname(self.path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        # 内部统一用 dict 存：更利于工程化（时间、热度、去重）
        self.memories: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self.memories = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except Exception:
            self.memories = []
            return

        # 兼容旧格式：list[str] -> list[dict]
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
            now = time.time()
            self.memories = []
            for x in data:
                t = _norm_text(x)
                if t:
                    self.memories.append(
                        {"text": t, "created_at": now, "updated_at": now, "count": 1}
                    )
            return

        cleaned: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = _norm_text(item.get("text", ""))
                if not text:
                    continue
                cleaned.append(
                    {
                        "text": text,
                        "created_at": float(item.get("created_at", time.time())),
                        "updated_at": float(item.get("updated_at", time.time())),
                        "count": int(item.get("count", 1)),
                    }
                )
        self.memories = cleaned[: self.max_items]

    def _atomic_save(self) -> None:
        """原子写入，避免半截 JSON。"""
        dir_ = os.path.dirname(self.path) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=".memory_", suffix=".json", dir=dir_, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)  # Windows/macOS/Linux 都可用
        finally:
            # 如果 replace 失败，清理临时文件（成功则 tmp_path 不存在）
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def add_memory(self, m: str) -> None:
        text = _norm_text(m)
        if not text:
            return
        now = time.time()

        # 归一化去重：重复则更新热度/时间
        for item in self.memories:
            if _norm_text(item.get("text", "")) == text:
                item["updated_at"] = now
                item["count"] = int(item.get("count", 1)) + 1
                self._atomic_save()
                return

        # 新记忆放前面
        self.memories.insert(
            0, {"text": text, "created_at": now, "updated_at": now, "count": 1}
        )
        if len(self.memories) > self.max_items:
            self.memories = self.memories[: self.max_items]
        self._atomic_save()

    def list_memories(self) -> List[str]:
        return [m["text"] for m in self.memories]

    def clear(self) -> None:
        self.memories = []
        self._atomic_save()

    def search(self, query: str, top_k: int = 5, min_score: float = 0.12) -> List[str]:
        """
        返回与 query 最相关的记忆（默认 top5）。
        关键点：没命中就返回 []，避免把无关记忆塞进 prompt。
        """
        q = _norm_text(query)
        if not q:
            return []

        qg = _ngrams(q, n=2)
        scored: List[Tuple[float, str]] = []
        for item in self.memories:
            text = item["text"]
            score = _jaccard(qg, _ngrams(text, n=2))
            if score >= min_score:
                scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]