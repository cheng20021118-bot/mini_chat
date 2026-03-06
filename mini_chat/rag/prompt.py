from typing import List, Optional, Dict

class PromptBuilder:
    @staticmethod
    def build_rag_prompt(docs: List[str]) -> str:
        if not docs:
            return "没有检索到资料。\n你必须回答：我不知道，资料中没有相关信息。"

        numbered = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
        return f"""
你是严格的问答系统。

【规则】
1. 只能使用资料中的信息
2. 不允许使用常识或外部知识
3. 资料没有就回答：我不知道，资料中没有相关信息。
4. 回答末尾必须给出引用编号，格式示例：引用：[1][3]

【资料】
{numbered}
""".strip()

    @staticmethod
    def build_memory_prompt(memory: str) -> Optional[Dict[str, str]]:
        if not memory:
            return None
        return {"role": "system", "content": f"以下是用户长期记忆，仅影响回答风格：\n{memory}"}
