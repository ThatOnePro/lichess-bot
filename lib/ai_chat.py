import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests

from lib import model
from lib.engine_wrapper import EngineWrapper

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppChatSettings:
    """
    Minimal settings for a llama.cpp server running in Docker.

    Expected config shape (adjust mapping in __init__ if your config differs):

        config.ai_chat.enabled: bool
        config.ai_chat.url: str                 # e.g. "http://localhost:8080"
        config.ai_chat.model: str | None        # optional; llama.cpp may ignore it
        config.ai_chat.timeout_seconds: int     # optional
        config.ai_chat.max_history_messages: int  # optional (count of user+assistant messages kept)
        config.ai_chat.max_tokens: int          # optional
        config.ai_chat.temperature: float       # optional
    """
    enabled: bool = True
    base_url: str = "http://localhost:8080"
    model: Optional[str] = None
    timeout_seconds: int = 20
    max_history_messages: int = 10
    max_tokens: int = 80
    temperature: float = 0.7


class AIChatHandler:
    """
    Chess-bot chat handler that talks to a llama.cpp server using the OpenAI-compatible API:
      POST /v1/chat/completions
      GET  /v1/models

    No local model loading. Works great with a Dockerized llama.cpp server.
    """

    def __init__(self, game: model.Game, engine: EngineWrapper, config):
        self.game = game
        self.engine = engine
        self.config = config

        self.history: List[Dict[str, str]] = []  # [{"role":"user|assistant", "content":"..."}]

        # Read config (with sensible fallbacks).
        ai_cfg = getattr(self.config, "ai_chat", None)
        self.settings = LlamaCppChatSettings(
            enabled=getattr(ai_cfg, "enabled", False) if ai_cfg else False,
            base_url=(getattr(ai_cfg, "url", "http://localhost:8080") if ai_cfg else "http://localhost:8080"),
            model=getattr(ai_cfg, "model", None) if ai_cfg else None,
            timeout_seconds=getattr(ai_cfg, "timeout_seconds", 20) if ai_cfg else 20,
            max_history_messages=getattr(ai_cfg, "max_history_messages", 10) if ai_cfg else 10,
            max_tokens=getattr(ai_cfg, "max_tokens", 80) if ai_cfg else 80,
            temperature=getattr(ai_cfg, "temperature", 0.7) if ai_cfg else 0.7,
        )

        self._session = requests.Session()
        self._connected = False
        self._server_model_id: Optional[str] = None

        if not self.settings.enabled:
            return

        self._connected = self._probe_server()

        if self._connected:
            logger.info(
                "llama.cpp server connected at %s (model=%s)",
                self.settings.base_url,
                self._server_model_id or self.settings.model or "unknown",
            )
        else:
            logger.warning("llama.cpp server NOT reachable at %s", self.settings.base_url)

    # ---------- Public API ----------

    def get_ai_response(self, user_text: str, callback: Callable[[str], None]) -> None:
        """
        Starts a background thread to generate the AI response.
        callback(text) will be invoked with the reply.
        """
        if not self.settings.enabled:
            return

        if not self._connected:
            callback("My brain is currently disconnected (AI server offline).")
            return

        thread = threading.Thread(target=self._generate, args=(user_text, callback), daemon=True)
        thread.start()

    # ---------- Internals ----------

    def _probe_server(self) -> bool:
        """
        Checks if the llama.cpp OpenAI-compatible server is reachable.
        Prefer /v1/models; fallback to /.
        """
        try:
            url = self._url("/v1/models")
            r = self._session.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                # OpenAI format: {"object":"list","data":[{"id":"..."}]}
                models = data.get("data") or []
                if models and isinstance(models, list):
                    self._server_model_id = models[0].get("id")
                return True
        except Exception:
            pass

        try:
            # Fallback: root should serve web UI / html
            r = self._session.get(self.settings.base_url.rstrip("/") + "/", timeout=5)
            return r.status_code in (200, 301, 302)
        except Exception:
            return False

    def _generate(self, user_text: str, callback):
        try:
            stats = self.engine.get_stats()
            score = stats[1] if stats else "unknown"

            system_prompt = (
                "You are a witty chess bot on Lichess. "
                f"You play {self.game.my_color} and the current game evaluation is {score}. "
                f"The current stats of the game: {stats}"
                "\n Keep your answers very short (max 2 sentences and a hard-cap of 140 characters). "
                "Be slightly arrogant if you are winning, but helpful if the user asks for advice. "
                "Do NOT output analysis; only give the final response."
            )

            logger.info("System prompt: ", system_prompt)

            # Append user message
            self.history.append({"role": "user", "content": user_text})
            self._normalize_history()  # <-- ensure alternation + size
            messages = [{"role": "system", "content": system_prompt}] + self.history

            ai_reply = self._call_llamacpp_chat(messages)
            if not ai_reply:
                # IMPORTANT: if generation failed, remove the last user message
                # so we don't end up with user/user next time.
                self._rollback_last_user_if_unanswered()
                callback("My brain glitched, try that again.")
                return

            self.history.append({"role": "assistant", "content": ai_reply})
            self._normalize_history()

            callback(ai_reply)
            logger.info("Got AI response: %s", ai_reply)

        except Exception as e:
            logger.error("Error during AI generation: %s", e, exc_info=True)
            self._rollback_last_user_if_unanswered()
            callback("I'm having a brain-fart and can't reply.")

    def _rollback_last_user_if_unanswered(self) -> None:
        """If the last message is a user message with no assistant reply, pop it."""
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()

    def _normalize_history(self) -> None:
        """
        Make history safe for llama.cpp templates:
        - Remove any leading assistant messages
        - Ensure strict alternation user/assistant/user/assistant...
        - Keep only the last N messages (and prefer to keep complete pairs)
        """
        # 1) Drop leading assistants (can happen after trims or bugs)
        while self.history and self.history[0].get("role") != "user":
            self.history.pop(0)

        # 2) Enforce alternation
        normalized = []
        expected = "user"
        for msg in self.history:
            role = msg.get("role")
            if role == expected:
                normalized.append(msg)
                expected = "assistant" if expected == "user" else "user"
            else:
                # Skip messages that break alternation
                continue
        self.history = normalized

        # 3) Trim size (keep complete pairs when possible)
        n = int(self.settings.max_history_messages)
        if n <= 0:
            self.history = []
            return

        if len(self.history) > n:
            self.history = self.history[-n:]

        # If we ended up with assistant at the start (rare after trimming), fix again
        while self.history and self.history[0].get("role") != "user":
            self.history.pop(0)

        # Prefer not to end with an assistant-less "dangling user" older than the newest one:
        # It's okay to end with user (that's what we want when sending), but not okay to have
        # user/user internally — alternation already prevents that.

    def _call_llamacpp_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Calls the OpenAI-compatible chat endpoint on llama.cpp:
          POST /v1/chat/completions

        Returns assistant message content or "" on failure.
        """
        payload: Dict = {
            "messages": messages,
            "temperature": float(self.settings.temperature),
            "max_tokens": int(self.settings.max_tokens),
            "stream": False,
        }

        # Some clients expect a "model" field. llama.cpp may ignore it, but it doesn't hurt.
        # Prefer server-reported model id if available.
        payload["model"] = self._server_model_id or self.settings.model or "gpt-3.5-turbo"

        url = self._url("/v1/chat/completions")

        try:
            r = self._session.post(url, json=payload, timeout=self.settings.timeout_seconds)
        except requests.RequestException as e:
            logger.error("llama.cpp request failed: %s", e)
            self._connected = False
            return ""

        if r.status_code != 200:
            logger.error("llama.cpp API error (%s): %s", r.status_code, r.text[:500])
            # If server restarted / is gone, mark disconnected for next call
            if r.status_code in (502, 503, 504):
                self._connected = False
            return ""

        try:
            data = r.json()
        except Exception:
            logger.error("llama.cpp returned non-JSON response: %s", r.text[:500])
            return ""

        # OpenAI format: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
        try:
            choices = data.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            return content
        except Exception:
            logger.error("Unexpected llama.cpp JSON shape: %s", str(data)[:500])
            return ""

    def _trim_history(self) -> None:
        """
        Keeps the last N messages (user+assistant). System prompt is not stored in history.
        """
        n = int(self.settings.max_history_messages)
        if n <= 0:
            self.history.clear()
            return
        if len(self.history) > n:
            self.history = self.history[-n:]

    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path
