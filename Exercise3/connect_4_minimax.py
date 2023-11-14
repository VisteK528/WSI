from typing import Tuple, List

from two_player_games.player import Player
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove, \
    ConnectFourState
from copy import copy
from random import choice
import numpy as np
import ipywidgets as widgets

ROW_COUNT = 6
COLUMN_COUNT = 7


class MinMaxSolver:

    def __init__(self, game: ConnectFour, row_count: int, column_count: int):
        self.game = game
        self._row_count = row_count
        self._column_count = column_count

        self._prizes = {
            "center": 4.,
            "plr_two_in_seg": 2.,
            "plr_three_in_seg": 10.,
            "plr_winning_cond": 1000.,
            "opp_two_in_seg": -2,
            "opp_winning_cond": -100
        }

    def evaluate_position(self, board: ConnectFourState,
                          player: Player) -> float:
        prize = 0
        # Center bonus
        center_column = board.fields[self._column_count // 2]
        center_count = center_column.count(player)
        prize += self._prizes["center"] * center_count

        # Check verticals
        for i in range(self._column_count):
            for j in range(self._row_count - 3):
                segment = [board.fields[i][j + z] for z in range(4)]
                prize += self._evaluate_segment(segment, player)

        # Check horizontals
        for i in range(self._column_count - 3):
            for j in range(self._row_count):
                segment = [board.fields[i + z][j] for z in range(4)]
                prize += self._evaluate_segment(segment, player)

        # Check rising edge diagonals
        for i in range(self._column_count - 3):
            for j in range(self._row_count - 3):
                segment = [board.fields[i + z][j + z] for z in range(4)]
                prize += self._evaluate_segment(segment, player)

        # Check falling edge diagonals
        for i in range(self._column_count - 3):
            for j in range(3, self._row_count):
                segment = [board.fields[i + z][j - z] for z in range(4)]
                prize += self._evaluate_segment(segment, player)

        return prize

    def _evaluate_segment(self, segment: List, player: Player):
        opponent = self.game.second_player
        if player == self.game.second_player:
            opponent = self.game.first_player

        prize = 0

        if segment.count(player) == 4:
            prize += self._prizes["plr_winning_cond"]
        elif segment.count(player) == 3 and segment.count(None) == 1:
            prize += self._prizes["plr_three_in_seg"]
        elif segment.count(player) == 2 and segment.count(None) == 2:
            prize += self._prizes["plr_two_in_seg"]
        elif segment.count(opponent) == 3 and segment.count(None) == 1:
            prize += self._prizes["opp_winning_cond"]
        """elif segment.count(opponent) == 2 and segment.count(None) == 2:
            prize += self._prizes["opp_two_in_seg"]"""

        return prize

    def is_valid_move(self, col_index: int) -> bool:
        if self.game.state.fields[col_index][-1] is None:
            return True
        else:
            return False

    def get_valid_moves(self):
        return [valid_column for valid_column in range(COLUMN_COUNT) if
                self.is_valid_move(valid_column)]

    def get_best_move(self, player: Player) -> int:
        valid_moves = self.get_valid_moves()

        best_prize = 0
        best_move = choice(valid_moves)

        for move in valid_moves:
            board_copy = copy(self.game.state)
            board_copy = board_copy.make_move(ConnectFourMove(move))
            prize = self.evaluate_position(board_copy, player)

            #print(f"Column: {move}\tPrize: {prize}")
            if prize > best_prize:
                best_prize = prize
                best_move = move

        #print(f"Chosen: {best_move}")
        return best_move

    def is_terminal(self) -> bool:
        # Check verticals
        for i in range(self._column_count):
            for j in range(self._row_count - 3):
                segment = [self.game.state.fields[i][j + z] for z in range(4)]
                if segment.count(self.game.first_player) == 4 or segment.count(
                        self.game.second_player) == 4:
                    return True

        # Check horizontals
        for i in range(self._column_count - 3):
            for j in range(self._row_count):
                segment = [self.game.state.fields[i + z][j] for z in range(4)]
                if segment.count(self.game.first_player) == 4 or segment.count(
                        self.game.second_player) == 4:
                    return True

        # Check rising edge diagonals
        for i in range(self._column_count - 3):
            for j in range(self._row_count - 3):
                segment = [self.game.state.fields[i + z][j + z] for z in
                           range(4)]
                if segment.count(self.game.first_player) == 4 or segment.count(
                        self.game.second_player) == 4:
                    return True

        # Check falling edge diagonals
        for i in range(self._column_count - 3):
            for j in range(3, self._row_count):
                segment = [self.game.state.fields[i + z][j - z] for z in
                           range(4)]
                if segment.count(self.game.first_player) == 4 or segment.count(
                        self.game.second_player) == 4:
                    return True

        # Check if the board is not completely full
        if sum([column.count(None) for column in game.state.fields]) == 0:
            return True

        return False

    def minimax(self, board, depth, alpha: float, beta: float,
                is_maximizing_player: bool) -> Tuple[int, float]:
        """Returns column index and score"""
        valid_moves = self.get_valid_moves()
        is_terminal = self.is_terminal()
        beta = (choice(valid_moves), beta)
        alpha = (choice(valid_moves), alpha)
        print(f"Initialization values: {alpha}\t{beta}\t{depth}")

        if is_terminal or depth == 0:
            if is_terminal:
                if self.game.get_winner() == self.game.first_player:
                    return (self.get_best_move(self.game.first_player), 100000)
                elif self.game.get_winner() == self.game.second_player:
                    return (self.get_best_move(self.game.second_player), -10000)
                else:
                    return (None, 0)
            else:
                return (self.get_best_move(self.game.second_player), self.evaluate_position(board, self.game.second_player))

        if is_maximizing_player:
            print("Max")
            for valid_move in valid_moves:
                board_copy = copy(board)
                board_copy = board_copy.make_move(ConnectFourMove(valid_move))

                new_alpha = self.minimax(board_copy, depth - 1, alpha[1], beta[1],
                                               not is_maximizing_player)

                print(f"Available alpha: {alpha}\t {new_alpha}\tDepth: {depth}")
                alpha = max(alpha, new_alpha, key=lambda x: x[1])
                print(f"Chosen alpha: {alpha}")

                if alpha[1] >= beta[1]:
                    break

            return alpha

        else:
            print("Min")
            for valid_move in valid_moves:
                board_copy = copy(board)
                board_copy = board_copy.make_move(ConnectFourMove(valid_move))

                new_beta = self.minimax(board_copy, depth - 1, alpha[1], beta[1],
                                         not is_maximizing_player)

                print(f"Available beta: {beta}\t {new_beta}\tDepth: {depth}")
                beta = min(beta, new_beta, key=lambda x: x[1])
                print(f"Chosen beta: {beta}")

                if alpha[1] >= beta[1]:
                    break
            return beta


if __name__ == "__main__":
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1,
                       second_player=p2)

    algorithm = MinMaxSolver(game, ROW_COUNT, COLUMN_COUNT)
    print(game)
    while not algorithm.is_terminal():
        chosen_column = input("Column: ")
        # print(f"Playing player: {p1.char}")
        game.make_move(ConnectFourMove(int(chosen_column)))

        print(game)

        print(f"Playing AI: {p2.char}")
        best_column, score = algorithm.minimax(algorithm.game.state, 2, -np.inf, np.inf, True)
        print(best_column, score)
        game.make_move(ConnectFourMove(best_column))

        print(game)

    print(f"Won: {game.get_winner().char}")
    print("Terminated")
