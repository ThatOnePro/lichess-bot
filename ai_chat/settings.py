from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaCppChatSettings:
    """
    Configuration for the llama.cpp AI chat feature.

    All values are read from the ``ai_chat`` section of config.yml:
        enabled, url, model, timeout_seconds, max_history_messages, max_tokens, temperature
    """
    enabled: bool = True
    base_url: str = "http://localhost:8080"
    model: Optional[str] = None
    timeout_seconds: int = 20
    max_history_messages: int = 10
    max_tokens: int = 80
    temperature: float = 0.7
