from typing import List, Dict

def get_recent_messages(messages: List[Dict], max_turns: int) -> List[Dict]:
    return messages[-max_turns:]

def messages_to_text(messages: List[Dict]) -> str:
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    return text

def compress_old_messages(client, model_name: str, messages: List[Dict]) -> str:
    text = messages_to_text(messages)
    prompt = f"""
请将以下对话压缩成一段简短摘要，保留关键事实和讨论结论，
用于未来对话参考，不要逐句复述。

{text}
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是对话压缩助手"},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content
