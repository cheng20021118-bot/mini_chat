from typing import List, Tuple

def should_reject(results: List[Tuple[float, str]], abs_th=0.42, gap_th=0.0) -> bool:
    """
    abs_th: top1 至少达到这个分数，否则拒答
    gap_th: top1-top2 差距太小也拒答（小库建议设 0）
    """
    if not results:
        return True
    top1 = results[0][0]
    if top1 < abs_th:
        return True
    if gap_th > 0 and len(results) >= 2:
        top2 = results[1][0]
        if (top1 - top2) < gap_th:
            return True
    return False
