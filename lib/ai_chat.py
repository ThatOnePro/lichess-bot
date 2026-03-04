import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import chess.engine
import requests

from lib import model
from lib.engine_wrapper import EngineWrapper

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppChatSettings:
    """
    Settings for an OpenAI-compatible llama.cpp server.

    Expected config keys under ``ai_chat``:
        enabled, url, model, timeout_seconds,
        max_history_messages, max_tokens, temperature
    """
    enabled: bool = True
    base_url: str = "http://localhost:8080"
    model: Optional[str] = None
    timeout_seconds: int = 20
    max_history_messages: int = 10
    max_tokens: int = 80
    temperature: float = 0.7


class AIChatHandler:
    """Chat handler that proxies to a llama.cpp server via POST /v1/chat/completions."""

    def __init__(self, game: model.Game, engine: EngineWrapper, config) -> None:
        self.game = game
        self.engine = engine
        self.history: List[Dict[str, str]] = []
        self._prev_score_cp: Optional[int] = None
        self._pending_player_move: Optional[tuple] = None

        ai_cfg = getattr(config, "ai_chat", None)

        def cfg(key: str, default):
            return getattr(ai_cfg, key, default) if ai_cfg else default

        self.settings = LlamaCppChatSettings(
            enabled=cfg("enabled", False),
            base_url=cfg("url", "http://localhost:8080"),
            model=cfg("model", None),
            timeout_seconds=cfg("timeout_seconds", 20),
            max_history_messages=cfg("max_history_messages", 10),
            max_tokens=cfg("max_tokens", 80),
            temperature=cfg("temperature", 0.7),
        )

        self._session = requests.Session()
        self._connected = False
        self._server_model_id: Optional[str] = None

        if not self.settings.enabled:
            return

        self._connected = self._probe_server()
        if self._connected:
            logger.info("llama.cpp connected at %s (model=%s)", self.settings.base_url,
                        self._server_model_id or self.settings.model or "unknown")
        else:
            logger.warning("llama.cpp NOT reachable at %s", self.settings.base_url)

    # ---------- Public API ----------

    def get_ai_response(self, user_text: str, callback: Callable[[str], None]) -> None:
        """Respond to a player chat message asynchronously via callback."""
        if not self.settings.enabled:
            return
        if not self._connected:
            callback("My brain is currently disconnected (AI server offline).")
            return
        threading.Thread(target=self._generate, args=(user_text, callback), daemon=True).start()

    @staticmethod
    def _describe_move(board: chess.Board, move_uci: str) -> str:
        """
        Returns a plain-English description of a move, e.g.:
          "Queen takes d5 (capturing a Rook)"
          "Knight to f3"
          "King-side castling"

        board must be the state BEFORE the move is applied so piece info is still present.
        """
        piece_names = {
            chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
            chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
        }
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return move_uci

        if move == chess.Move.from_uci("e1g1") or move == chess.Move.from_uci("e1c1") or \
           move == chess.Move.from_uci("e8g8") or move == chess.Move.from_uci("e8c8"):
            side = "King-side" if move.to_square in (chess.G1, chess.G8) else "Queen-side"
            return f"{side} castling"

        piece = board.piece_at(move.from_square)
        piece_name = piece_names.get(piece.piece_type, "Piece") if piece else "Piece"
        to_sq = chess.square_name(move.to_square)

        captured = board.piece_at(move.to_square)
        # en-passant: no piece on target square but pawn captures diagonally
        if not captured and piece and piece.piece_type == chess.PAWN and \
                chess.square_file(move.from_square) != chess.square_file(move.to_square):
            captured_name = "Pawn"
            return f"{piece_name} takes {to_sq} en passant (capturing a {captured_name})"

        if captured:
            captured_name = piece_names.get(captured.piece_type, "piece")
            return f"{piece_name} takes {to_sq} (capturing a {captured_name})"

        if move.promotion:
            promo_name = piece_names.get(move.promotion, "piece")
            return f"Pawn promotes to {promo_name} on {to_sq}"

        return f"{piece_name} to {to_sq}"

    def after_move(self, board: chess.Board, move_uci: str, mover_color: str,
                   send_message: Callable[[str], None]) -> None:
        """
        Triggered after every move. We evaluate the swing on the bot's turn because
        that's the first moment the engine has re-evaluated the position after the
        player moved. Comparing scores[-1] (now) vs scores[-2] (last bot move) tells
        us how much the player's previous move helped or hurt them.
        """
        if not self.settings.enabled or not self._connected:
            return

        bot_moved = (mover_color == str(self.game.my_color))
        if not bot_moved:
            # Store the board+move so we can describe the player's move once the bot replies
            self._pending_player_move = (board.copy(), move_uci)
            return

        # Bot just moved — engine.scores now has the fresh evaluation
        if len(self.engine.scores) < 2:
            return  # need at least two data points for a meaningful delta

        if self._pending_player_move is None:
            return

        player_board, player_uci = self._pending_player_move
        self._pending_player_move = None

        try:
            # Both scores are from the bot's POV; scores are appended after each bot search
            prev_cp = self.engine.scores[-2].relative.score(mate_score=3000)
            curr_cp = self.engine.scores[-1].relative.score(mate_score=3000)
        except Exception:
            return

        if prev_cp is None or curr_cp is None:
            return

        # Positive = position improved for bot (player blundered)
        # Negative = position worsened for bot (player made a strong move)
        delta = curr_cp - prev_cp

        # Reconstruct the board before the player's move to describe the piece that moved
        pre_move_board = player_board.copy()
        try:
            pre_move_board.pop()
        except Exception:
            pre_move_board = player_board

        move_desc = self._describe_move(pre_move_board, player_uci)
        eval_now = f"{curr_cp / 100:+.2f}"

        if delta >= 150:
            context = (
                f"Your opponent just played {move_desc} and blundered, handing you +{delta / 100:.1f} pawns "
                f"(eval now {eval_now}). Mock them specifically about what they just did — "
                "e.g. if they dropped a queen, say something about the queen. Be witty, not mean."
            )
        elif delta <= -100:
            context = (
                f"Your opponent just played {move_desc}, a strong move that cost you {abs(delta) / 100:.1f} pawns "
                f"(eval now {eval_now} for you). Acknowledge specifically what they did, but stay confident."
            )
        else:
            return

        threading.Thread(target=self._generate_move_comment, args=(context, send_message), daemon=True).start()

    # ---------- Internals ----------

    def _latest_engine_score_cp(self) -> Optional[int]:
        """
        Returns the latest score as centipawns from the bot's POV.
        Uses mate_score=3000 so forced mates don't produce None but a large finite value.
        """
        if not self.engine.scores:
            return None
        try:
            return self.engine.scores[-1].relative.score(mate_score=3000)
        except Exception:
            return None

    def _game_context(self) -> str:
        """Builds the shared personality/context block injected into every system prompt."""
        eval_str, winrate_str = "unknown", "unknown"
        if self.engine.scores:
            pov = self.engine.scores[-1]
            try:
                cp = pov.relative.score(mate_score=3000)
                eval_str = f"{cp / 100:+.2f}" if cp is not None else "unknown"
            except Exception:
                pass
            try:
                # ply estimate: each entry in engine.scores represents one bot move (every other half-move)
                ply = len(self.engine.scores) * 2
                winrate_str = f"{round(pov.wdl(model='sf', ply=ply).relative.expectation() * 100, 1)}%"
            except Exception:
                pass

        bot_moves_played = len(self.engine.scores)
        phase = "opening" if bot_moves_played < 7 else "middlegame" if bot_moves_played < 25 else "endgame"

        return (
            f"You are {self.game.me.name}, a cocky, sharp-tongued chess bot on Lichess "
            f"playing as {self.game.my_color} against {self.game.opponent.name}. "
            f"It is the {phase} (roughly move {bot_moves_played}). "
            f"Engine eval: {eval_str} pawns (your perspective). Win probability: {winrate_str}. "
            "Personality: confident, a little arrogant when winning, darkly amused by mistakes — "
            "grandmaster trash talk, never outright insults. "
            "Rules: ONE punchy sentence, hard cap 140 characters, no hashtags, no analysis notation."
        )

    def _generate_move_comment(self, context: str, send_message: Callable[[str], None]) -> None:
        """Generate and send an unprompted comment after a notable move."""
        try:
            messages = [
                {"role": "system", "content": self._game_context()},
                {"role": "user", "content": context},
            ]
            reply = self._call_llamacpp_chat(messages)
            if reply:
                send_message(reply)
                # logger.info("Move comment sent: %s", reply)
        except Exception:
            logger.error("Error generating move comment", exc_info=True)

    def _generate(self, user_text: str, callback: Callable[[str], None]) -> None:
        """Generate a reply to a player chat message, maintaining conversation history."""
        try:
            self.history.append({"role": "user", "content": user_text})
            self._normalize_history()

            stats = self.engine.get_stats()
            score = stats[1] if len(stats) > 1 else (stats[0] if stats else "unknown")
            system_prompt = (
                f"{self._game_context()} "
                f"Current stats: {', '.join(stats) if stats else 'none'}. "
                "Be helpful if the user asks for advice."
            )

            messages = [{"role": "system", "content": system_prompt}] + self.history
            reply = self._call_llamacpp_chat(messages)

            if not reply:
                self._rollback_last_user()
                callback("My brain glitched, try that again.")
                return

            self.history.append({"role": "assistant", "content": reply})
            self._normalize_history()
            callback(reply)
            logger.info("AI response: %s", reply)

        except Exception:
            logger.error("Error during AI generation", exc_info=True)
            self._rollback_last_user()
            callback("I'm having a brain-fart and can't reply.")

    def _rollback_last_user(self) -> None:
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()

    def _normalize_history(self) -> None:
        """
        Enforces strict user/assistant alternation and trims to max_history_messages.
        llama.cpp chat templates require perfectly alternating roles — any deviation
        can cause garbled output or a server error.
        """
        # Drop any leading assistant turns
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

        # Rebuild with strict alternation, dropping any out-of-order messages
        normalized, expected = [], "user"
        for msg in self.history:
            if msg["role"] == expected:
                normalized.append(msg)
                expected = "assistant" if expected == "user" else "user"
        self.history = normalized

        # Trim to size, then fix any leading assistant that appeared after slicing
        n = int(self.settings.max_history_messages)
        if n <= 0:
            self.history = []
            return
        if len(self.history) > n:
            self.history = self.history[-n:]
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

    def _probe_server(self) -> bool:
        """Check connectivity; also grab the first available model id if possible."""
        try:
            r = self._session.get(self._url("/v1/models"), timeout=5)
            if r.status_code == 200:
                models = r.json().get("data") or []
                if models:
                    self._server_model_id = models[0].get("id")
                return True
        except Exception:
            pass
        try:
            return self._session.get(self.settings.base_url.rstrip("/") + "/", timeout=5).status_code in (200, 301, 302)
        except Exception:
            return False

    def _call_llamacpp_chat(self, messages: List[Dict[str, str]]) -> str:
        """POST to /v1/chat/completions and return the assistant content, or '' on failure."""
        payload = {
            "model": self._server_model_id or self.settings.model or "gpt-3.5-turbo",
            "messages": messages,
            "temperature": float(self.settings.temperature),
            "max_tokens": int(self.settings.max_tokens),
            "stream": False,
        }
        try:
            r = self._session.post(self._url("/v1/chat/completions"), json=payload,
                                   timeout=self.settings.timeout_seconds)
        except requests.RequestException as e:
            logger.error("llama.cpp request failed: %s", e)
            self._connected = False
            return ""

        if r.status_code != 200:
            logger.error("llama.cpp API error (%s): %s", r.status_code, r.text[:500])
            if r.status_code in (502, 503, 504):
                self._connected = False
            return ""

        try:
            choices = r.json().get("choices", [])
            return (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        except Exception:
            logger.error("Unexpected llama.cpp response: %s", r.text[:500])
            return ""

    def _url(self, path: str) -> str:
        return self.settings.base_url.rstrip("/") + path
