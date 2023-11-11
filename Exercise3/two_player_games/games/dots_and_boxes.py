from typing import Dict, Iterable, List, Optional, Tuple
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class DotsAndBoxes(Game):
    """Class that represents the dots and boxes game"""
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, size: int = 2, first_player: Player = None, second_player: Player = None):
        """
        Initializes game.

        Parameters:
            size: the size of the game as number of columns and rows of boxes
            first_player: the player that will go first (if None is passed, a player will be created)
            second_player: the player that will go second (if None is passed, a player will be created)
        """
        self.first_player = first_player or Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        self.second_player = second_player or Player(self.SECOND_PLAYER_DEFAULT_CHAR)

        state = DotsAndBoxesState(self.first_player, self.second_player, size)

        super().__init__(state)


class DotsAndBoxesMove(Move):
    """
    Class that represents a move in the dots and boxes game

    Variables:
        connection: str, 'h' if a horizontal line or 'v' if a vertical line
        loc: line coordinates as a tuple: (column, row) for horizontal or (row, column) for vertical
    """
    def __init__(self, connection: str, loc: Tuple[int, int]):
        self.connection = connection
        self.loc = loc

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DotsAndBoxesMove):
            return False
        return self.connection == o.connection and self.loc == o.loc


class DotsAndBoxesState(State):
    """Class that represents a state in the dots and boxes game"""
    def __init__(self,
            current_player: Player, other_player: Player, size: int = None,
            horizontals: List[List[bool]] = None, verticals: List[List[bool]] = None, boxes: List[List[Player]] = None):
        """Creates the state. Do not call directly."""

        if horizontals and verticals and boxes:
            self.horizontals = horizontals
            self.verticals = verticals
            self.boxes = boxes
        elif size:
            self.horizontals = [[False for _ in range(size + 1)] for _ in range(size)]
            self.verticals = [[False for _ in range(size + 1)] for _ in range(size)]
            self.boxes = [[None for _ in range(size)] for _ in range(size)]
        else:
            raise ValueError("Cannot initialize state, parameters missing")

        super().__init__(current_player, other_player)

    def get_moves(self) -> Iterable[DotsAndBoxesMove]:
        return [
            DotsAndBoxesMove("h", loc)
            for loc in self._get_free_lines(self.horizontals)
        ] + [
            DotsAndBoxesMove("v", loc)
            for loc in self._get_free_lines(self.verticals)
        ]

    def make_move(self, move: DotsAndBoxesMove) -> 'DotsAndBoxesState':
        collection = self.horizontals if move.connection == "h" else self.verticals
        if collection[move.loc[0]][move.loc[1]]:
            raise ValueError("Invalid move")

        horizontals = self._set(self.horizontals, move.loc) if move.connection == "h" else self.horizontals
        verticals = self._set(self.verticals, move.loc) if move.connection == "v" else self.verticals
        boxes, changed = self._check_boxes_after_move(horizontals, verticals, move)

        if changed:
            next_player = self._current_player
            other_player = self._other_player
        else:
            next_player = self._other_player
            other_player = self._current_player

        return DotsAndBoxesState(
            next_player, other_player,
            horizontals=horizontals, verticals=verticals, boxes=boxes
        )

    def is_finished(self) -> bool:
        return all(all(box_row) for box_row in self.boxes)

    def get_winner(self) -> Optional[Player]:
        if not self.is_finished():
            return None
        scores = self.get_scores()
        if scores[self._current_player] > scores[self._other_player]:
            return self._current_player
        elif scores[self._current_player] < scores[self._other_player]:
            return self._other_player
        else:
            return None

    def __str__(self) -> str:
        text = []

        for row in range(len(self.boxes)):
            text.append(self._lines_row_to_str(row))
            text.append(self._row_to_str(row))

        text.append(self._lines_row_to_str(len(self.boxes)))

        return f'Current player: {self._current_player.char}\n' + '\n'.join(text)

    def get_scores(self) -> Dict[Player, int]:
        scores = {
            self._current_player: 0,
            self._other_player: 0
        }

        for row in self.boxes:
            for box in row:
                if box:
                    scores[box] += 1

        return scores

    # below are helper methods for the public interface

    def _get_free_lines(self, collection: List[List[bool]]) -> List[Tuple[int, int]]:
        return [
            (loc1, loc2)
            for loc1, subcol in enumerate(collection)
            for loc2, line in enumerate(subcol)
            if not line
        ]

    def _set(self, collection: List[List[bool]], loc: Tuple[int, int]) -> List[List[bool]]:
        return [
            [
                True if loc2 == loc[1] else line
                for loc2, line in enumerate(subcol)
            ] if loc1 == loc[0] else subcol
            for loc1, subcol in enumerate(collection)
        ]

    def _check_box(self, horizontals, verticals, col, row) -> Player:
        if self.boxes[row][col]:
            return self.boxes[row][col]

        if horizontals[col][row] and horizontals[col][row + 1] and verticals[row][col] and verticals[row][col + 1]:
            return self._current_player

        return None

    def _check_boxes_after_move(
            self, horizontals: List[List[bool]], verticals: List[List[bool]],
            move: DotsAndBoxesMove) -> Tuple[List[List[Player]], bool]:
        box1_col = None
        box1_row = None        

        box2_col = None
        box2_row = None

        if move.connection == "h":
            col, row = move.loc
            if row > 0:
                box1_col = col
                box1_row = row - 1
            if row < len(self.boxes):
                box2_col = col
                box2_row = row
        else:
            row, col = move.loc
            if col > 0:
                box1_col = col - 1
                box1_row = row
            if col < len(self.boxes[0]):
                box2_col = col
                box2_row = row

        new_boxes = [
            [
                self._check_box(horizontals, verticals, col, row)
                if col in (box1_col, box2_col) else box
                for col, box in enumerate(row_boxes)
            ] if row in (box1_row, box2_row) else row_boxes
            for row, row_boxes in enumerate(self.boxes)
        ]

        changed = False
        if box1_col is not None:
            if self.boxes[box1_row][box1_col] != new_boxes[box1_row][box1_col]:
                changed = True

        if box2_col is not None:
            if self.boxes[box2_row][box2_col] != new_boxes[box2_row][box2_col]:
                changed = True

        return new_boxes, changed

    def _lines_row_to_str(self, row):
        return 'o' + 'o'.join('-' if col[row] else ' ' for col in self.horizontals) + 'o'

    def _row_to_str(self, row):
        chars = []
        for line, box in zip(self.verticals[row], self.boxes[row]):
            chars.append('|' if line else ' ')
            chars.append(box.char if box else ' ')

        chars.append('|' if self.verticals[row][-1] else ' ')

        return ''.join(chars)
