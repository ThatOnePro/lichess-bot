import logging
import re
import threading
from typing import Callable, Optional

import chess

from lib import model
from lib.engine_wrapper import EngineWrapper
from .context_provider import build_coaching_context, build_trash_talk_context, game_phase
from .history import ChatHistory
from .move_describer import describe_move
from .server_client import LlamaCppClient
from .settings import LlamaCppChatSettings

logger = logging.getLogger(__name__)

_LICHESS_CHAT_LIMIT = 140


def _split_for_chat(text: str, limit: int = _LICHESS_CHAT_LIMIT) -> list[str]:
    """
    Split a long reply into chunks that fit within Lichess's chat message limit.
    Splits at sentence boundaries so each chunk reads naturally.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current_chunk = ""
    for sentence in raw_sentences:
        candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        if len(candidate) <= limit:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If a single sentence exceeds the limit, hard-cut it
            while len(sentence) > limit:
                chunks.append(sentence[:limit])
                sentence = sentence[limit:]
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks if chunks else [text[:limit]]


class AIChatHandler:
    """
    Coordinates AI chat during a Lichess game.

    Two entry points:
    - ``get_ai_response``: reply to a message typed by the opponent.
    - ``after_move``: optionally comment after a notable move (blunder or strong move).
    """

    def __init__(self, game: model.Game, engine: EngineWrapper, config) -> None:
        self.game = game
        self.engine = engine
        self._pending_player_move: Optional[tuple] = None

        ai_cfg = getattr(config, "ai_chat", None)

        def cfg(key, default):
            return getattr(ai_cfg, key, default) if ai_cfg else default

        settings = LlamaCppChatSettings(
            enabled=cfg("enabled", False),
            base_url=cfg("url", "http://localhost:8080"),
            model=cfg("model", None),
            timeout_seconds=cfg("timeout_seconds", 20),
            max_history_messages=cfg("max_history_messages", 10),
            max_tokens=cfg("max_tokens", 80),
            coaching_max_tokens=cfg("coaching_max_tokens", 350),
            temperature=cfg("temperature", 0.7),
        )

        self._enabled = settings.enabled
        self._client = LlamaCppClient(settings)
        self._history = ChatHistory(settings.max_history_messages)
        self._coaching_max_tokens = settings.coaching_max_tokens
        self._latest_board: Optional[chess.Board] = None
        self._last_player_move_desc: Optional[str] = None

        if not self._enabled:
            return

        self._client.probe()
        if self._client.connected:
            logger.info(
                "llama.cpp connected at %s (model=%s)",
                settings.base_url,
                self._client.model_id or settings.model or "unknown",
            )
        else:
            logger.warning("llama.cpp NOT reachable at %s", settings.base_url)

    # Public API

    def get_ai_response(self, user_text: str, callback: Callable[[str], None]) -> None:
        """Reply to a player chat message asynchronously via callback."""
        if not self._enabled:
            return
        if not self._client.connected:
            callback("My brain is currently disconnected (AI server offline).")
            return
        threading.Thread(target=self._generate, args=(user_text, callback), daemon=True).start()

    def after_move(self, board: chess.Board, move_uci: str, mover_color: str,
                   send_message: Callable[[str], None]) -> None:
        """
        Called after every move. When the bot just moved, score the quality of
        the opponent's previous move and optionally send a trash-talk or compliment.
        """
        if not self._enabled or not self._client.connected:
            return

        bot_moved = (mover_color == str(self.game.my_color))
        if not bot_moved:
            # Save the player's move so we can evaluate it once the bot replies
            self._pending_player_move = (board.copy(), move_uci)
            return

        # Bot just moved — keep the latest board for coaching context
        self._latest_board = board.copy()

        # We now have a fresh engine evaluation to compare against the player's last move
        if len(self.engine.scores) < 2 or self._pending_player_move is None:
            return

        player_board, player_uci = self._pending_player_move
        self._pending_player_move = None

        try:
            prev_cp = self.engine.scores[-2].relative.score(mate_score=3000)
            curr_cp = self.engine.scores[-1].relative.score(mate_score=3000)
        except Exception:
            return

        if prev_cp is None or curr_cp is None:
            return

        # Positive delta = bot gained (player blundered); negative = bot lost (player played well)
        delta = curr_cp - prev_cp

        if abs(delta) < 75:
            return  # Move was not notable enough to comment on

        # player_board is the state AFTER the player moved; pop it to describe the piece used
        pre_move_board = player_board.copy()
        try:
            pre_move_board.pop()
        except Exception:
            pre_move_board = player_board

        move_desc = describe_move(pre_move_board, player_uci)
        self._last_player_move_desc = move_desc
        context = build_trash_talk_context(self.game, self.engine, board, move_desc, delta)

        threading.Thread(target=self._generate_move_comment, args=(context, send_message), daemon=True).start()

    # Internals

    def _trash_talk_personality(self) -> str:
        """Build the personality block injected as the system prompt for move comments."""
        move_num = len(self.engine.scores)
        phase = game_phase(move_num)
        return (
            f"You are {self.game.me.name}, a cocky, sharp-tongued chess bot on Lichess "
            f"playing as {self.game.my_color} against {self.game.opponent.name}. "
            f"It is the {phase} (roughly move {move_num}). "
            "Personality: confident, a little arrogant when winning, darkly amused by mistakes — "
            "grandmaster trash talk, never outright insults. "
            "Rules: ONE punchy sentence, hard cap 140 characters, no hashtags, no analysis notation."
        )

    def _generate(self, user_text: str, callback: Callable[[str], None]) -> None:
        """Generate a reply to a player chat message (runs in a background thread)."""
        try:
            self._history.add("user", user_text)
            board = self._latest_board or chess.Board()
            coaching_ctx = build_coaching_context(
                self.game, self.engine, board, self._last_player_move_desc
            )
            messages = [{"role": "system", "content": coaching_ctx}] + self._history.messages
            reply = self._client.chat(messages, max_tokens=self._coaching_max_tokens)

            if not reply:
                self._history.rollback_last_user()
                callback("My brain glitched, try that again.")
                return

            self._history.add("assistant", reply)
            logger.info("AI response: %s", reply)
            for chunk in _split_for_chat(reply):
                callback(chunk)

        except Exception:
            logger.error("Error during AI generation", exc_info=True)
            self._history.rollback_last_user()
            callback("I'm having a brain-fart and can't reply.")

    def _generate_move_comment(self, context: str, send_message: Callable[[str], None]) -> None:
        """Generate and send an unprompted comment after a notable move (background thread)."""
        try:
            messages = [
                {"role": "system", "content": self._trash_talk_personality()},
                {"role": "user", "content": context},
            ]
            reply = self._client.chat(messages)
            if reply:
                send_message(reply)
        except Exception:
            logger.error("Error generating move comment", exc_info=True)
