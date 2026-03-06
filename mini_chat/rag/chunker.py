# rag/chunker.py
from __future__ import annotations

def split_text(text: str, chunk_size: int = 420, overlap: int = 80) -> list[str]:
    """
    简单稳健的切块：
    - chunk_size: 每块字符数
    - overlap: 相邻块重叠字符数（帮助跨句信息检索）
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 过滤极短文本
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= 20:  # 过滤太短碎片
            chunks.append(chunk)
        start += step

    # 去重（保持顺序）
    seen = set()
    dedup = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            dedup.append(c)

    return dedup
