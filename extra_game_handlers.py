"""Functions for the user to implement when the config file is not adequate to express bot requirements."""
from lib import model
from lib.lichess_types import OPTIONS_TYPE
import chess
import logging


def game_specific_options(game: model.Game) -> OPTIONS_TYPE:  # noqa: ARG001
    """
    Return a dictionary of engine options based on game aspects.

    By default, an empty dict is returned so that the options in the configuration file are used.
    """
    return {}


def is_supported_extra(challenge: model.Challenge) -> bool:  # noqa: ARG001
    """
    Determine whether to accept a challenge.

    By default, True is always returned so that there are no extra restrictions beyond those in the config file.
    """
    return True


def after_move(
    game: model.Game, board: chess.Board, move_uci: str, mover_color: str
) -> None:  # noqa: ARG001
    """
    Hook called after every new move (player or bot).

    :param game: The current game state.
    :param board: The board after the move has been applied.
    :param move_uci: The move in UCI format.
    :param mover_color: "white" or "black" indicating who made the move.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "after_move called: game=%r board=%r move_uci=%s mover_color=%s",
        game,
        board,
        move_uci,
        mover_color,
    )
    return
