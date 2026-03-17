"""
Builds rich English context strings from game and engine state for the AI chat system.

Two modes:
- build_trash_talk_context : focused on the opponent's last move quality (brief, punchy).
  Result is used as the "user" message in the move-comment generation prompt.
- build_coaching_context   : comprehensive game + position state for answering player questions.
  Result is injected into the system prompt so the LLM has all information it needs.

Neither function knows anything about chess tactics — it just translates engine numbers
and board state into plain English so the LLM can reason about them.
"""
from typing import Optional

import chess

from lib import model
from lib.engine_wrapper import EngineWrapper
from .move_describer import describe_move

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# Rough material values in pawns (king excluded from balance)
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def game_phase(move_num: int) -> str:
    if move_num < 7:
        return "opening"
    if move_num < 25:
        return "middlegame"
    return "endgame"


def _format_eval(engine: EngineWrapper, player_is_bot: bool = False) -> str:
    """
    Current engine evaluation as a plain-English string.

    ``engine.scores[-1].relative`` is always from the bot's perspective (positive = bot winning).
    When the player asking is the human opponent (player_is_bot=False), we invert the sign
    so "positive = you are winning" refers to the human player.
    """
    if not engine.scores:
        return "unknown"
    try:
        score = engine.scores[-1].relative
        cp = score.score()
        if cp is None:
            mate = score.mate()
            if mate is None:
                return "unknown"
            # Positive mate = bot has forced mate; invert for human player
            if not player_is_bot:
                mate = -mate
            direction = "for you" if mate > 0 else "against you"
            return f"forced checkmate in {abs(mate)} moves ({direction})"
        # Invert for the human player perspective
        if not player_is_bot:
            cp = -cp
        if cp == 0:
            return "equal (0.00 pawns)"
        sign = "you are winning by" if cp > 0 else "you are losing by"
        return f"{sign} {abs(cp) / 100:.2f} pawns ({cp / 100:+.2f})"
    except Exception:
        return "unknown"


def _format_winrate(engine: EngineWrapper, player_is_bot: bool = False) -> str:
    """
    Win probability as a plain-English percentage.

    When player_is_bot=False the percentage is inverted so it reflects the human player's chances.
    """
    if not engine.scores:
        return "unknown"
    try:
        ply = len(engine.scores) * 2
        bot_pct = round(engine.scores[-1].wdl(model="sf", ply=ply).relative.expectation() * 100, 1)
        pct = bot_pct if player_is_bot else round(100 - bot_pct, 1)
        if pct > 66:
            quality = "strong winning chances"
        elif pct > 55:
            quality = "slight advantage"
        elif pct >= 45:
            quality = "roughly equal"
        elif pct >= 34:
            quality = "slight disadvantage"
        else:
            quality = "strong losing chances"
        return f"{pct}% win probability for you ({quality})"
    except Exception:
        return "unknown"


def _material_balance(board: chess.Board, my_color: chess.Color) -> str:
    """Plain-English material balance from *my* perspective."""
    my_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, my_color))
        for pt in _PIECE_VALUES
    )
    opp_mat = sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, not my_color))
        for pt in _PIECE_VALUES
    )
    diff = my_mat - opp_mat
    if diff > 0:
        return f"You are up {diff} material point(s)"
    if diff < 0:
        return f"You are down {abs(diff)} material point(s)"
    return "Material is equal between both sides"


def _pieces_on_board(board: chess.Board) -> str:
    """Inventory of remaining pieces for both sides."""
    parts = []
    for color, label in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        items = []
        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            count = len(board.pieces(pt, color))
            if count:
                name = _PIECE_NAMES[pt] + ("s" if count > 1 else "")
                items.append(f"{count} {name}")
        parts.append(f"{label}: {', '.join(items) if items else 'King only'}")
    return "; ".join(parts)


def _best_continuation(engine: EngineWrapper) -> Optional[str]:
    """
    Return the engine's suggested best continuation as a SAN move string
    (already converted by engine_wrapper.add_comment).
    """
    if not engine.move_commentary:
        return None
    info = engine.move_commentary[-1]
    # "ponderpv" is the full variation in SAN; "Pv" is the same after normalisation
    pv = info.get("ponderpv") or info.get("Pv")
    if not pv:
        return None
    # Return only the first 3 half-moves to stay readable
    tokens = pv.split()[:3]
    return " ".join(tokens) if tokens else None


def _recent_move_history(board: chess.Board, last_n: int = 6) -> str:
    """
    Describe the last *last_n* half-moves in plain English.
    Returns an empty string when no moves have been played.
    """
    if not board.move_stack:
        return "No moves have been played yet."

    # Collect last N moves and reconstruct the board states they started from
    all_moves = list(board.move_stack)
    slice_start = max(0, len(all_moves) - last_n)
    recent_moves = all_moves[slice_start:]

    # Rewind the board to just before those moves
    tmp = board.copy()
    for _ in recent_moves:
        try:
            tmp.pop()
        except Exception:
            break

    lines = []
    for move in recent_moves:
        color_label = "White" if tmp.turn == chess.WHITE else "Black"
        full_move = tmp.fullmove_number
        desc = describe_move(tmp, move.uci())
        lines.append(f"Move {full_move} {color_label}: {desc}")
        tmp.push(move)

    return "; ".join(lines)


def _active_threats(board: chess.Board, my_color: chess.Color) -> list[str]:
    """List notable positional threats visible on the board."""
    threats: list[str] = []

    if board.is_check():
        checker_color = "White" if board.turn == chess.BLACK else "Black"
        threats.append(
            f"The {'White' if board.turn == chess.WHITE else 'Black'} king is in check (attacked by {checker_color})")

    # Hanging pieces (attacked but not defended) — check both sides
    for color, label in [(my_color, "Your"), (not my_color, "Opponent's")]:
        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for sq in board.pieces(pt, color):
                attacked_by_opp = board.is_attacked_by(not color, sq)
                defended = board.is_attacked_by(color, sq)
                if attacked_by_opp and not defended:
                    name = _PIECE_NAMES[pt].title()
                    threats.append(f"{label} {name} on {chess.square_name(sq)} is hanging (attacked and undefended)")
                elif attacked_by_opp:
                    name = _PIECE_NAMES[pt].title()
                    threats.append(f"{label} {name} on {chess.square_name(sq)} is under attack (but defended)")

    return threats[:6]  # Cap to keep context concise


def _last_move_quality_sentence(engine: EngineWrapper, player_is_bot: bool = False) -> str:
    """
    Translate the last evaluation swing into a plain-English quality sentence.

    engine.scores deltas are from the bot's perspective. When player_is_bot=False we invert
    so "good move" and "blunder" are relative to the human player.
    """
    if len(engine.scores) < 2:
        return ""
    try:
        prev_cp = engine.scores[-2].relative.score(mate_score=3000)
        curr_cp = engine.scores[-1].relative.score(mate_score=3000)
        if prev_cp is None or curr_cp is None:
            return ""
        # Positive delta = bot gained centipawns = player lost centipawns
        bot_delta = curr_cp - prev_cp
        delta = bot_delta if player_is_bot else -bot_delta
        if delta >= 200:
            return f"The last move was excellent — you gained {abs(delta) / 100:.1f} pawns of advantage."
        if delta >= 75:
            return f"The last move was a good move — you gained {abs(delta) / 100:.1f} pawns."
        if delta <= -200:
            return f"The last move was a blunder — the evaluation swung {abs(delta) / 100:.1f} pawns against you."
        if delta <= -75:
            return f"The last move was an inaccuracy — you lost {abs(delta) / 100:.1f} pawns of advantage."
        return "The last move was roughly neutral (evaluation barely changed)."
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_trash_talk_context(
        game: model.Game,
        engine: EngineWrapper,
        board: chess.Board,
        move_desc: str,
        delta_cp: int,
) -> str:
    """
    Build the context string passed as the "user" message for trash-talk move comments.

    :param board:     Current board position (after the bot's reply move).
    :param move_desc: Plain-English description of the opponent's last move.
    :param delta_cp:  Centipawn delta from bot's perspective (positive = opponent blundered).
    """
    move_num = len(engine.scores)
    phase = game_phase(move_num)
    eval_now = _format_eval(engine)
    my_color = chess.WHITE if game.my_color == "white" else chess.BLACK

    if delta_cp >= 200:
        quality = f"a serious blunder (handed you +{delta_cp / 100:.1f} pawns)"
        tone = "Mock them specifically about what they just did. Be witty and cutting."
    elif delta_cp >= 75:
        quality = f"an inaccuracy (gave you +{delta_cp / 100:.1f} pawns)"
        tone = "Tease them about the slip. Stay clever and specific."
    elif delta_cp <= -100:
        quality = f"a strong move (cost you {abs(delta_cp) / 100:.1f} pawns)"
        tone = "Acknowledge the good play sarcastically but stay confident you'll recover."
    else:
        quality = "a neutral move"
        tone = "Make a light remark about the game state."

    material = _material_balance(board, my_color)

    return (
        f"Game phase: {phase} (approx. move {move_num}). "
        f"Your opponent just played: {move_desc} — this was {quality}. "
        f"Engine evaluation right now: {eval_now}. "
        f"{material}. "
        f"{tone} "
        "ONE punchy sentence, hard cap 140 characters, no hashtags, no move notation."
    )


def build_coaching_context(
        game: model.Game,
        engine: EngineWrapper,
        board: chess.Board,
        player_last_move_desc: Optional[str] = None,
) -> str:
    """
    Build a comprehensive game-state context block injected into the system prompt
    when the bot is answering a coaching question from the player.

    All perspective markers ("you", "your") refer to the HUMAN PLAYER (the opponent of the bot),
    not the bot itself. The LLM is acting as a coach for the human, not as the bot.

    :param player_last_move_desc: Plain-English description of the human player's last move
                                  (the one before the bot responded). Clarifies which move
                                  belongs to the human vs the bot's most recent reply.

    The LLM does not understand chess rules or notation on its own — this function
    translates everything into plain English so it can reason and answer helpfully.
    """
    move_num = len(engine.scores)
    phase = game_phase(move_num)
    turn_label = "White" if board.turn == chess.WHITE else "Black"

    # Human player is the opponent of the bot
    player_color = chess.WHITE if game.opponent_color == "white" else chess.BLACK

    # All perspective-sensitive helpers use player_is_bot=False (human player's POV)
    eval_now = _format_eval(engine, player_is_bot=False)
    winrate = _format_winrate(engine, player_is_bot=False)
    last_move_quality = _last_move_quality_sentence(engine, player_is_bot=False)

    material = _material_balance(board, player_color)
    pieces = _pieces_on_board(board)
    recent_history = _recent_move_history(board)
    threats = _active_threats(board, player_color)
    best_line = _best_continuation(engine)

    threats_str = (
        "\n".join(f"  - {t}" for t in threats)
        if threats
        else "  - No immediate threats or hanging pieces detected"
    )
    # The engine's best line is from the bot's side; clarify this for the LLM
    best_line_str = (
        f"Engine's suggested best continuation for the bot ('{game.me.name}'): {best_line}. "
        f"This means the bot plans to play those moves — knowing this can help you anticipate and prepare."
        if best_line
        else "No engine continuation available for this position."
    )
    last_move_str = last_move_quality if last_move_quality else "Move quality data not available."

    # Rating info (may be None for AI opponents)
    bot_rating = f" (rated {game.me.rating})" if game.me.rating else ""
    player_rating = f" (rated {game.opponent.rating})" if game.opponent.rating else ""

    # Explicitly label the last two half-moves so the LLM cannot confuse them
    if player_last_move_desc:
        last_moves_block = (
            "=== THE LAST TWO HALF-MOVES ===\n"
            f"  1. Human player ('{game.opponent.name}', {game.opponent_color}) played: {player_last_move_desc}.\n"
            f"     → This was the human player's move. They moved their own piece.\n"
            f"  2. Bot ('{game.me.name}', {game.my_color}) then responded.\n"
            f"     → The most recent entry in the move history below is the BOT's response, not the human's.\n"
            f"  {last_move_str}\n"
        )
    else:
        last_moves_block = (
            "=== LAST MOVE QUALITY ===\n"
            f"  {last_move_str}\n"
            f"  Note: The most recent move in the history below is the BOT's move, "
            f"not the human player's.\n"
        )

    return (
        "=== CHESS GAME CONTEXT ===\n"
        f"You are coaching the human player '{game.opponent.name}'{player_rating} "
        f"who is playing as {game.opponent_color}.\n"
        f"Their opponent is the chess bot '{game.me.name}'{bot_rating} playing as {game.my_color}.\n"
        f"Game type: {game.perf_name}, {game.mode}.\n"
        "IMPORTANT: In this entire context, 'you' and 'your' always refer to the human player "
        f"('{game.opponent.name}', playing as {game.opponent_color}), never to the bot.\n"
        "\n"
        "=== CURRENT POSITION ===\n"
        f"Game phase: {phase} (approximately move {move_num}).\n"
        f"It is {turn_label}'s turn to move next.\n"
        f"Pieces remaining — {pieces}.\n"
        f"{material}.\n"
        "\n"
        "=== ENGINE EVALUATION (from your perspective as the human player) ===\n"
        "The engine is a program that calculates the best chess moves. Its numbers, translated for you:\n"
        f"  - Current evaluation: {eval_now}.\n"
        f"    (Positive means YOU, the human player, are ahead; negative means the bot is ahead;\n"
        f"     one 'pawn' of advantage is a meaningful edge in chess.)\n"
        f"  - {winrate}.\n"
        f"  - {best_line_str}\n"
        "\n"
        f"{last_moves_block}"
        "\n"
        "=== THREATS & CHECKS ===\n"
        f"{threats_str}\n"
        "\n"
        "=== RECENT MOVE HISTORY (most recent entry = bot's last move) ===\n"
        f"{recent_history}\n"
        "\n"
        "=== YOUR ROLE AS A COACH ===\n"
        f"You are a gentle, encouraging chess coach helping '{game.opponent.name}' (the human player, {game.opponent_color}). "
        "Use ONLY the information above to answer their question. Explain in plain English — if you mention "
        "a chess term, immediately clarify what it means. Never be condescending. Be concise but thorough. "
        "Do NOT trash-talk. When referring to moves or pieces, always clarify which side they belong to. "
        "Keep your answer to 2-3 sentences."
    )
