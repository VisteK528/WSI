from typing import Iterable, Optional
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class Game:
    """Game interface."""
    def __init__(self, state: State):
        """
        Initializes game.

        Parameters:
            state: initial game state
        """
        self.state = state

    def get_moves(self) -> Iterable[Move]:
        """
        Returns:
            Possible moves
        """
        return self.state.get_moves()

    def get_current_player(self) -> Player:
        """
        Returns:
            Current player
        """
        return self.state.get_current_player()

    def make_move(self, move: Move):
        """
        Makes move.

        Parameters:
            move: move to make
        """
        self.state = self.state.make_move(move)

    def is_finished(self) -> bool:
        """
        Returns:
            If the game is finished
        """
        return self.state.is_finished()

    def get_winner(self) -> Optional[Player]:
        """
        Returns:
            Player that is the winner or None if not finished or draw
        """
        return self.state.get_winner()

    def get_players(self) -> Iterable[Player]:
        """
        Returns:
            List of players
        """
        return self.state.get_players()

    def __str__(self) -> str:
        """
        Returns:
            Printable text represenation of the game's state
        """
        return str(self.state)
