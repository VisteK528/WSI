from typing import Tuple, List
from two_player_games.player import Player
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove, \
    ConnectFourState
from copy import copy
from random import choice
import numpy as np
import time


def simulate_move(board, column):
    board_copy = copy(board)
    board_copy = board_copy.make_move(ConnectFourMove(column))
    return board_copy


class Solver:
    def __init__(self, game: ConnectFour, row_count: int, column_count: int, max_player: Player, min_player: Player):
        self.game = game
        self._row_count = row_count
        self._column_count = column_count

        self._max_player = max_player
        self._min_player = min_player

    def is_valid_move(self, col_index: int) -> bool:
        # Check if move to column is available
        if self.game.state.fields[col_index][-1] is None:
            return True
        else:
            return False

    def get_valid_moves(self):
        # Return list of all currently available moves
        return [valid_column for valid_column in range(self._column_count) if
                self.is_valid_move(valid_column)]

    def get_best_move(self, *args):
        pass


class RandomSolver(Solver):
    def __init__(self, game: ConnectFour, row_count: int, column_count: int, max_player: Player, min_player: Player):
        super().__init__(game, row_count, column_count, max_player, min_player)

    def get_best_move(self, *args):
        return choice(self.get_valid_moves())


class HeuristicSolver(Solver):
    def __init__(self, game: ConnectFour, row_count: int, column_count: int, max_player: Player, min_player: Player):
        super().__init__(game, row_count, column_count, max_player, min_player)

        self._prizes = {
            "plr_two_in_seg": 5,
            "plr_three_in_seg": 10,
            "plr_winning_cond": 20,
            "opp_winning_cond": 80
        }

        total_weight = sum(self._prizes.values())
        self._prizes = {key: value / total_weight for key, value in
                        self._prizes.items()}

    def evaluate_position(self, board: ConnectFourState,
                          player: Player) -> float:
        prize = 0

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
        prize = 0

        # Evaluate max player and min player possibility
        # for winning combination in the future
        if segment.count(self._max_player) == 1 and segment.count(None) == 3:
            prize += 1
        elif segment.count(self._min_player) == 1 and segment.count(None) == 3:
            prize -= 1

        # Evaluate already existing combinations
        if segment.count(self._max_player) == 4:
            prize += self._prizes["plr_winning_cond"]
        elif segment.count(self._max_player) == 3 and segment.count(None) == 1:
            prize += self._prizes["plr_three_in_seg"]
        elif segment.count(self._max_player) == 2 and segment.count(None) == 2:
            prize += self._prizes["plr_two_in_seg"]
        elif segment.count(self._min_player) == 3 and segment.count(None) == 1:
            prize -= self._prizes["opp_winning_cond"]

        return prize

    def get_best_move(self, *args):
        scores = []
        for move in self.get_valid_moves():
            board_copy = simulate_move(self.game.state, move)
            scores.append((move, self.evaluate_position(board_copy, self._max_player)))

        moves = [score[0] for score in scores if score[1] == max([score[1] for score in scores])]
        return choice(moves)


class MinMaxSolver(HeuristicSolver):
    def __init__(self, game: ConnectFour, row_count: int, column_count: int,
                 max_player: Player, min_player: Player):
        super().__init__(game, row_count, column_count, max_player, min_player)

    def get_best_move(self, depth, alpha_beta=True, *args):
        if alpha_beta:
            col, score = self.minimax_alpha_beta(self.game.state, depth,
                                                 -np.inf, np.inf,
                                                 True)
            return col
        else:
            col, score = self.minimax(self.game.state, depth, True)
            return col

    def minimax_alpha_beta(self, board, depth, alpha: float, beta: float,
                is_maximizing_player: bool) -> Tuple[int, float]:
        # Returns column index and score
        valid_moves = self.get_valid_moves()
        is_terminal = board.is_finished()

        if is_terminal or depth == 0:
            if is_terminal:
                if board.get_winner() == self._max_player:
                    return (None, 1e10)
                elif board.get_winner() == self._min_player:
                    return (None, -1e10)
                else:
                    return (None, 0)
            else:
                return (
                    None, self.evaluate_position(board, self._max_player))

        if is_maximizing_player:
            value = -np.inf
            chosen_column = choice(valid_moves)
            for valid_move in valid_moves:
                evaluation = self.minimax_alpha_beta(simulate_move(board, valid_move), depth - 1, alpha, beta, False)[1]

                if evaluation > value:
                    value = evaluation
                    chosen_column = valid_move

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return chosen_column, value

        else:
            value = np.inf
            chosen_column = choice(valid_moves)
            for valid_move in valid_moves:
                evaluation = self.minimax_alpha_beta(simulate_move(board, valid_move), depth - 1, alpha, beta, True)[1]

                if evaluation < value:
                    value = evaluation
                    chosen_column = valid_move

                beta = min(beta, value)

                if alpha >= beta:
                    break

            return chosen_column, value

    def minimax(self, board, depth, is_maximizing_player: bool) -> Tuple[int, float]:
        """Returns column index and score"""
        valid_moves = self.get_valid_moves()
        is_terminal = board.is_finished()

        if is_terminal or depth == 0:
            if is_terminal:
                if board.get_winner() == self._max_player:
                    return (None, 1e10)
                elif board.get_winner() == self._min_player:
                    return (None, -1e10)
                else:
                    return (None, 0)
            else:
                return (
                    None, self.evaluate_position(board, self._max_player))

        evaluations = []
        for valid_move in valid_moves:
            evaluation = self.minimax(simulate_move(board, valid_move), depth - 1, not is_maximizing_player)[1]
            evaluations.append((valid_move, evaluation))

        if is_maximizing_player:
            moves = [score for score in evaluations if
                     score[1] == max([score[1] for score in evaluations])]

            return choice(moves)

        else:
            moves = [score for score in evaluations if
                     score[1] == min([score[1] for score in evaluations])]

            return choice(moves)
