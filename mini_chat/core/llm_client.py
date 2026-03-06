from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_stream(self, messages):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
