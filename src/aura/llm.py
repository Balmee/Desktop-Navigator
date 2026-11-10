from ollama import Client
from aura.config import settings

class AuraLLM:
    def __init__(self):
        self.client = Client(host=settings.ollama_host)
        self.messages = [{"role": "system", "content": settings.system_prompt}]

    def check_connection(self):
        return self.client.list()

    def chat(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat(
            model=settings.model,
            messages=self.messages,
            stream=False,
        )
        answer = response.message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def ask_about_image(self, prompt: str, image_path: str) -> str:
        # The image is included in this request only, not retained in chat history.
        messages = self.messages + [{
            "role": "user",
            "content": prompt,
            "images": [image_path],
        }]
        response = self.client.chat(
            model=settings.model,
            messages=messages,
            stream=False,
        )
        answer = response.message.content or ""
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": answer})
        return answer
