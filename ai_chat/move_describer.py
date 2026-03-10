import chess

_PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}


def describe_move(board: chess.Board, move_uci: str) -> str:
    """
    Return a plain-English description of a move.

    ``board`` must be the position BEFORE the move is applied, so piece info is available.

    Examples:
        "Queen takes d5 (capturing a Rook)"
        "Knight to f3"
        "King-side castling"
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return move_uci

    # Castling
    castling_moves = {
        chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1"),
        chess.Move.from_uci("e8g8"), chess.Move.from_uci("e8c8"),
    }
    if move in castling_moves:
        side = "King-side" if move.to_square in (chess.G1, chess.G8) else "Queen-side"
        return f"{side} castling"

    piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(piece.piece_type, "Piece") if piece else "Piece"
    to_sq = chess.square_name(move.to_square)
    captured = board.piece_at(move.to_square)

    # En passant: pawn captures diagonally but target square is empty
    if (not captured and piece and piece.piece_type == chess.PAWN
            and chess.square_file(move.from_square) != chess.square_file(move.to_square)):
        return f"{piece_name} takes {to_sq} en passant (capturing a Pawn)"

    if captured:
        captured_name = _PIECE_NAMES.get(captured.piece_type, "piece")
        return f"{piece_name} takes {to_sq} (capturing a {captured_name})"

    if move.promotion:
        promo_name = _PIECE_NAMES.get(move.promotion, "piece")
        return f"Pawn promotes to {promo_name} on {to_sq}"

    return f"{piece_name} to {to_sq}"
