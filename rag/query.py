import re

STOP_PATTERNS = [
    r"是什么", r"是啥", r"什么意思", r"定义", r"介绍一下", r"解释一下",
    r"请问", r"帮我", r"给我", r"一下", r"吗", r"\?", r"？"
]

def normalize_query(q: str) -> str:
    q0 = (q or "").strip()
    q = q0
    for p in STOP_PATTERNS:
        q = re.sub(p, "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or q0
