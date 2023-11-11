from typing import Dict, Iterable, List, Optional, Tuple
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class Morris(Game):
    """Class that represents the morris games. Current implementation does not allow flying."""
    def __init__(
            self, n_pawns: int, size: int, connections: List[Tuple[int, int]],
            possible_morrises: List[Tuple[int, int, int]], moves_limit: int,
            grid_str: str, first_player: Player, second_player: Player) -> None:
        self.first_player = first_player
        self.second_player = second_player

        state = MorrisState(
            n_pawns, size, connections, possible_morrises, moves_limit, 
            grid_str, first_player, second_player
        )

        super().__init__(state)


class SixMensMorris(Morris):
    """Class that represents the Six Men's Morris Game."""
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, first_player: Player = None, second_player: Player = None) -> None:
        n_pawns = 6
        size = 16
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),  # outer
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 8),  # inner
            (1, 9), (3, 11), (5, 13), (7, 15)  # between outer and inner
        ]
        possible_morrises = [
            (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0),  # outer
            (8, 9, 10), (10, 11, 12), (12, 13, 14), (14, 15, 8)  # inner
        ]
        moves_limit = 40
        grid_str = (
            "[{0}]-----[{1}]-----[{2}]\n"
            + " |       |       |\n"
            + " |  [{8}]-[{9}]-[{10}]  |\n"
            + " |   |       |   |\n"
            + "[{7}]-[{15}]     [{11}]-[{3}]\n"
            + " |   |       |   |\n"
            + " |  [{14}]-[{13}]-[{12}]  |\n"
            + " |       |       |\n"
            + "[{6}]-----[{5}]-----[{4}]\n"
        )

        if first_player is None:
            first_player = Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        if second_player is None:
            second_player = Player(self.SECOND_PLAYER_DEFAULT_CHAR)

        super().__init__(
            n_pawns, size, connections, possible_morrises,
            moves_limit, grid_str, first_player, second_player
        )


class MorrisMove(Move):
    """Class that represents a move in the morris game.

    It has 3 numerical fields:
     - take_pawn - represents field from which a move is made. Optional.
     - place_pawn - represents field to which a pawn is placed. Required.
     - remove_pawn - represents field from which an opponent's pawn is removed. Optional.

    There are 4 possible move types:
     - placing pawn to place_pawn
     - move pawn from take_pawn to place_pawn
     - additionally, an enemy pawn may be removed
    """
    def __init__(
            self, take_pawn: Optional[int] = None,
            place_pawn: Optional[int] = None,
            remove_pawn: Optional[int] = None) -> None:
        self.take_pawn = take_pawn
        self.place_pawn = place_pawn
        self.remove_pawn = remove_pawn

        super().__init__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MorrisMove):
            return False
        return (
            self.take_pawn == o.take_pawn and self.place_pawn == o.place_pawn
            and self.remove_pawn == o.remove_pawn
        )


class MorrisState(State):
    """Represents a state in the morris game. Current implementation does not allow flying."""
    def __init__(
            self, n_pawns: int, size: int, connections: List[Tuple[int, int]],
            possible_morrises: List[Tuple[int, int, int]], moves_limit: int,
            grid_str: str, current_player: Player, other_player: Player,
            n_moves: int = None, placed_pawns: Dict[Player, int] = None,
            grid: List[Optional[Player]] = None) -> None:
        self.n_pawns = n_pawns
        self.size = size
        self.connections = connections
        self.possible_morrises = possible_morrises
        self.moves_limit = moves_limit

        self.grid_str = grid_str

        if n_moves is None or placed_pawns is None or grid is None:
            n_moves = 0
            placed_pawns = {current_player: 0, other_player: 0}
            grid = [None for _ in range(size)]

        self.n_moves = n_moves
        self.placed_pawns = placed_pawns
        self.grid = grid

        super().__init__(current_player, other_player)

        self.finished, self.winner = self.check_finished()

    def check_finished(self) -> Tuple[bool, Optional[Player]]:
        pawns = {
            self._current_player: self.n_pawns - self.placed_pawns[self._current_player],
            self._other_player: self.n_pawns - self.placed_pawns[self._other_player]
        }

        for field in self.grid:
            if field is not None:
                pawns[field] += 1

        if pawns[self._current_player] < 3:
            return True, self._other_player
        elif pawns[self._other_player] < 3:
            return True, self._current_player

        available_move = False
        if self.placed_pawns[self._current_player] < self.n_pawns:
            for field in self.grid:
                if field is None:
                    available_move = True
                    break
        for connection in self.connections:
            if (
                self.grid[connection[0]] is self._current_player and self.grid[connection[1]] is None
                or self.grid[connection[1]] is self._current_player and self.grid[connection[0]] is None
            ):
                available_move = True
                break

        if not available_move:
            return True, self._other_player

        elif self.n_moves == self.moves_limit:
            return True, None

        return False, None

    def get_moves(self) -> Iterable[MorrisMove]:
        if self.placed_pawns[self._current_player] < self.n_pawns:
            moves = [(None, i) for i, field in enumerate(self.grid) if field is None]
        else:
            moves = []
            for connection in self.connections:
                if self.grid[connection[0]] is self._current_player and self.grid[connection[1]] is None:
                    moves.append((connection[0], connection[1]))
                if self.grid[connection[1]] is self._current_player and self.grid[connection[0]] is None:
                    moves.append((connection[1], connection[0]))

        other_player_pawns = {i for i, field in enumerate(self.grid) if field == self._other_player}
        other_player_pawns_in_morrises = set()
        for morris in self.possible_morrises:
            if all(self.grid[i] == self._other_player for i in morris):
                other_player_pawns_in_morrises.update(morris)

        other_player_pawns_not_in_morrises = other_player_pawns.difference(other_player_pawns_in_morrises)

        if other_player_pawns_not_in_morrises:
            removables = other_player_pawns_not_in_morrises
        else:
            removables = other_player_pawns_in_morrises

        moves_list = []

        for move in moves:
            makes_morris = False
            for morris in self.possible_morrises:
                if move[0] not in morris and move[1] in morris:
                    if len([i for i in morris if self.grid[i] == self._current_player]) == 2:
                        makes_morris = True
                        break

            if makes_morris:
                moves_list.extend(MorrisMove(*move, removable) for removable in removables)
            else:
                moves_list.append(MorrisMove(*move, None))

        return moves_list

    def make_move(self, move: MorrisMove) -> 'MorrisState':
        if self.finished:
            raise ValueError("Cannot make move on finished game")
        new_grid = list(self.grid)
        n_moves = self.n_moves
        new_placed_pawns = dict(self.placed_pawns)

        if move.take_pawn is not None:
            if self.grid[move.take_pawn] is not self._current_player:
                raise ValueError("Cannot move pawn from empty space")
            new_grid[move.take_pawn] = None
        else:
            if self.placed_pawns[self._current_player] == self.n_pawns:
                raise ValueError("Maximum pawns number already placed")
            new_placed_pawns[self._current_player] += 1

        if self.grid[move.place_pawn] is not None:
            raise ValueError("Cannot place pawn at occupied space")
        new_grid[move.place_pawn] = self._current_player

        if move.remove_pawn is not None:
            if self.grid[move.remove_pawn] is not self._other_player:
                raise ValueError("Cannot remove pawn from empty space")
            new_grid[move.remove_pawn] = None
            n_moves = 0
        else:
            n_moves += 1

        return MorrisState(
            self.n_pawns, self.size, self.connections, self.possible_morrises, self.moves_limit, self.grid_str,
            self._other_player, self._current_player, n_moves, new_placed_pawns, new_grid
        )

    def is_finished(self) -> bool:
        return self.finished

    def get_winner(self) -> Optional[Player]:
        return self.winner

    def __str__(self) -> str:
        if self.is_finished():
            current_player_text = ""
            if self.get_winner() is None:
                finished_text = "Draw!"
            else:
                finished_text = "Winner: Player " + self.get_winner().char
        else:
            current_player_text = "Current player: " + self._current_player.char
            finished_text = ""

        if all(v == self.n_pawns for v in self.placed_pawns.values()):
            pawns_to_place = ""
        else:
            pawns_to_place = (
                f"Pawns to place:\n\tplayer {self._current_player.char}: {self.n_pawns - self.placed_pawns[self._current_player]}"
                + f"\tplayer {self._other_player.char}: {self.n_pawns - self.placed_pawns[self._other_player]}\n"
            )

        return (
            self.grid_str.format(*[' ' if field is None else field.char for field in self.grid])
            + pawns_to_place + current_player_text + finished_text
        )
