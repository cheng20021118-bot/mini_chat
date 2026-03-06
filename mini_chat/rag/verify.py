from typing import List

def verify_answer(client, model_name: str, answer: str, docs: List[str]) -> bool:
    """
    统一成 4 参：client, model_name, answer, docs
    这样你 app 里调用不会再出现 “3参/4参”对不上。
    """
    context = "\n".join(docs)
    prompt = f"""
请判断下面的回答是否完全可以从资料中推出。

资料：
{context}

回答：
{answer}

如果回答包含资料中没有的信息，请回答：不合格
如果完全基于资料，请回答：合格
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return "合格" in resp.choices[0].message.content
