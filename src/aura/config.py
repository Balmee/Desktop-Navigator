from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("AURA_OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("AURA_MODEL", "qwen3-vl:8b")
    system_prompt: str = os.getenv(
        "AURA_SYSTEM_PROMPT",
        "You are AURA, a helpful, careful, locally operated desktop AI companion. "
        "Be clear, practical, and honest about uncertainty."
    )

settings = Settings()
