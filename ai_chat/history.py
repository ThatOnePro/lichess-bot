from typing import Dict, List


class ChatHistory:
    """
    Manages the conversation history sent to the AI.

    llama.cpp chat templates require strictly alternating user/assistant turns,
    so this class enforces that invariant on every update.
    """

    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self._messages: List[Dict[str, str]] = []

    @property
    def messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def add(self, role: str, content: str) -> None:
        """Append a message and re-normalize the history."""
        self._messages.append({"role": role, "content": content})
        self._normalize()

    def rollback_last_user(self) -> None:
        """Remove the last user message (used when the AI fails to reply)."""
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()

    def _normalize(self) -> None:
        """Enforce strict alternation and trim to max_messages."""
        # History must start with a user turn
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)

        # Keep only messages that follow the correct user -> assistant -> user -> ... pattern
        normalized, expected = [], "user"
        for msg in self._messages:
            if msg["role"] == expected:
                normalized.append(msg)
                expected = "assistant" if expected == "user" else "user"
        self._messages = normalized

        # Trim oldest messages if over the limit
        if self.max_messages <= 0:
            self._messages = []
            return
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

        # After trimming, ensure we still start with a user turn
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)
