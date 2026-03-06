# memory/extractor.py
from typing import List

class MemoryExtractor:
    @staticmethod
    def extract(client, prompt: str,model_name) -> List[str]:
        """
        client: OpenAI(...) 客户端
        """
        schema_prompt = f"""
你是长期记忆抽取器。
从用户输入中抽取 1~2 条稳定、可复用的信息（偏好/背景/目标），不要抽取通用知识。
如果没有可抽取信息，返回空列表。

用户输入：
{prompt}

请输出 JSON 数组，例如：
["用户是华中科技大学计算机研究生", "用户希望回答更详细"]
没有则输出：[]
"""

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个信息抽取器，只输出 JSON 数组。"},
                {"role": "user", "content": schema_prompt},
            ],
        )

        text = resp.choices[0].message.content.strip()

        # 容错：如果模型输出不是纯 JSON，尽量截取 []
        import re, json
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []

        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            return []

        return []
