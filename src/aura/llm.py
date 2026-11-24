
from ollama import Client
from aura.config import settings


class AuraLLM:
    def __init__(self):
        self.client = Client(host=settings.ollama_host)
        self.messages = [
            {"role": "system", "content": settings.system_prompt}
        ]

    def check_connection(self):
        return self.client.list()

    def chat(self, prompt: str) -> str:
        """Normal conversational interaction with persistent history."""

        self.messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=settings.model,
            messages=self.messages,
            stream=False,
        )

        answer = response.message.content or ""
        self.messages.append(
            {"role": "assistant", "content": answer}
        )

        return answer

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate an independent response without modifying
        AURA's normal conversational history.
        """

        system_message = system_prompt or settings.system_prompt

        response = self.client.chat(
            model=settings.model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=False,
        )

        return response.message.content or ""

    def ask_about_image(self, prompt: str, image_path: str) -> str:
        """Ask a question about an image using the normal conversation."""

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

        self.messages.append(
            {"role": "user", "content": prompt}
        )
        self.messages.append(
            {"role": "assistant", "content": answer}
        )

        return answer
