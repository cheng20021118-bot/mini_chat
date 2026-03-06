"""Intent routing for the Streamlit chat app.

We distinguish 3 intents:

- MEMORY_QUERY: user is asking about what we remember.
- MEMORY_WRITE: user is telling stable personal info / preferences to remember.
- KB_QA: default; use RAG over the knowledge base.

This keeps the strict RAG gate (to avoid hallucination) while making the app
usable for user-profile inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class Intent(str, Enum):
    MEMORY_QUERY = "memory_query"
    MEMORY_WRITE = "memory_write"
    KB_QA = "kb_qa"


# --- Heuristics ---

_MEMORY_QUERY_PATTERNS = [
    r"你(还)?记得我(吗)?",
    r"我是谁",
    r"关于我(的)?(信息|资料)",
    r"(我的)?(目标|偏好|喜好|爱好|背景|身份)是什么",
    r"我(的)?(目标|偏好|喜好|爱好|背景|身份)是什么",
]

# Users telling stable info / preferences. We keep this broad, and then
# require NOT a question to route to MEMORY_WRITE.
_MEMORY_WRITE_MARKERS = [
    "我叫",
    "我是",
    "我的目标",
    "目标是",
    "我希望",
    "我偏好",
    "我喜欢",
    "我爱",
    "我讨厌",
    "我不喜欢",
    "从现在开始",
    "记住",
    "请记住",
]

_QUESTION_PATTERNS = [
    r"\?$",
    r"？$",
    r"(是什么|是啥|什么意思|定义|解释|介绍一下|怎么|如何|为什么|多少|能不能|可不可以)",
    r"吗$",
]


def is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # explicit question marks
    if "?" in t or "？" in t:
        return True
    # heuristic question words
    for p in _QUESTION_PATTERNS:
        if re.search(p, t):
            return True
    return False


def is_memory_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(re.search(p, t) for p in _MEMORY_QUERY_PATTERNS)


def is_memory_write(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(m in t for m in _MEMORY_WRITE_MARKERS)


def route_intent(text: str) -> Intent:
    """Route user input to an intent.

    Priority:
    1) MEMORY_QUERY
    2) MEMORY_WRITE (only if not a question)
    3) KB_QA
    """
    if is_memory_query(text):
        return Intent.MEMORY_QUERY
    if is_memory_write(text) and not is_question(text):
        return Intent.MEMORY_WRITE
    return Intent.KB_QA
