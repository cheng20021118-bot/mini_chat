import re
from typing import List, Tuple

def extract_keywords(query: str) -> List[str]:
    if not query:
        return []
    parts = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)
    seen = set()
    kws = []
    for p in parts:
        p = p.strip().lower()
        if len(p) <= 1:
            continue
        if p not in seen:
            seen.add(p)
            kws.append(p)
    return kws

def select_docs_with_fallback(
    results: List[Tuple[float, str]],
    query: str,
    top_sim_threshold: float,
    min_keep: int = 1
) -> List[str]:
    """
    1) 阈值过滤
    2) 过滤后为空 -> 保留 top1
    3) 词面关键词命中 -> rerank 提升“主题一致性”
    """
    if not results:
        return []

    # 先阈值过滤
    filtered = [(s, d) for (s, d) in results if s >= top_sim_threshold]
    if not filtered:
        filtered = [results[0]]  # 兜底保留 top1

    kws = extract_keywords(query)
    if not kws:
        return [d for _, d in filtered]

    def bonus(doc: str) -> int:
        low = doc.lower()
        return sum(1 for kw in kws if kw in low)

    reranked = sorted(filtered, key=lambda x: (bonus(x[1]), x[0]), reverse=True)

    # 强制保留包含关键词的
    must = [d for s, d in reranked if bonus(d) > 0]

    final_docs = []
    for d in must:
        if d not in final_docs:
            final_docs.append(d)

    # 补齐至少 min_keep
    need = max(min_keep, len(final_docs))
    for s, d in reranked:
        if len(final_docs) >= need:
            break
        if d not in final_docs:
            final_docs.append(d)

    if not final_docs:
        final_docs = [results[0][1]]
    return final_docs
