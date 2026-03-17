import logging
from typing import Dict, List, Optional

import requests

from .settings import LlamaCppChatSettings

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """Handles all HTTP communication with the llama.cpp OpenAI-compatible server."""

    def __init__(self, settings: LlamaCppChatSettings) -> None:
        self.settings = settings
        self._session = requests.Session()
        self.connected = False
        self.model_id: Optional[str] = None

    def probe(self) -> bool:
        """Check server connectivity and discover the available model ID."""
        try:
            r = self._session.get(self._url("/v1/models"), timeout=5)
            if r.status_code == 200:
                models = r.json().get("data") or []
                if models:
                    self.model_id = models[0].get("id")
                self.connected = True
                return True
        except Exception:
            pass

        # Fall back to a plain GET on the root
        try:
            self.connected = self._session.get(
                self.settings.base_url.rstrip("/") + "/", timeout=5
            ).status_code in (200, 301, 302)
        except Exception:
            self.connected = False

        return self.connected

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """
        Send a chat completion request and return the assistant's reply.
        Returns an empty string on any failure.

        :param max_tokens: Override the default max_tokens from settings (e.g. for coaching replies).
        """
        payload = {
            "model": self.model_id or self.settings.model or "gpt-3.5-turbo",
            "messages": messages,
            "temperature": float(self.settings.temperature),
            "max_tokens": int(max_tokens if max_tokens is not None else self.settings.max_tokens),
            "stream": False,
        }
        try:
            r = self._session.post(
                self._url("/v1/chat/completions"),
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as e:
            logger.error("llama.cpp request failed: %s", e)
            self.connected = False
            return ""

        if r.status_code != 200:
            logger.error("llama.cpp API error (%s): %s", r.status_code, r.text[:500])
            if r.status_code in (502, 503, 504):
                self.connected = False
            return ""

        try:
            choices = r.json().get("choices", [])
            return (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        except Exception:
            logger.error("Unexpected llama.cpp response: %s", r.text[:500])
            return ""

    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path
