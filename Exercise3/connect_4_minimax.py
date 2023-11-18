import time
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
            "plr_two_in_seg": 1e2,
            "plr_three_in_seg": 1e4,
            "plr_winning_cond": 1e10,
            "opp_two_in_seg": 1,
            "opp_winning_cond": 1e6
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

                # Ostateczność
                """if player == self.game.first_player:
                    if segment.count(player) == 2 and segment.count(None) == 2:
                        prize += self._prizes["plr_two_in_seg"] / 2
                else:
                    if segment.count(player) == 2 and segment.count(None) == 2:
                        prize -= self._prizes["plr_two_in_seg"] / 2"""

        # Check falling edge diagonals
        for i in range(self._column_count - 3):
            for j in range(3, self._row_count):
                segment = [board.fields[i + z][j - z] for z in range(4)]
                prize += self._evaluate_segment(segment, player)

                # Ostateczność
                """if player == self.game.first_player:
                    if segment.count(player) == 2 and segment.count(None) == 2:
                        prize += self._prizes["plr_two_in_seg"] / 2
                else:
                    if segment.count(player) == 2 and segment.count(None) == 2:
                        prize -= self._prizes["plr_two_in_seg"] / 2"""

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
            prize -= self._prizes["opp_winning_cond"]
        elif segment.count(opponent) == 2 and segment.count(None) == 2:
            prize -= self._prizes["opp_two_in_seg"]

        return prize

    def is_valid_move(self, col_index: int) -> bool:
        """
        Check if move to column is available
        """
        if self.game.state.fields[col_index][-1] is None:
            return True
        else:
            return False

    def get_valid_moves(self):
        """
        Return list of all currently available moves
        """
        return [valid_column for valid_column in range(COLUMN_COUNT) if
                self.is_valid_move(valid_column)]

    def get_best_move_2(self, depth: int) -> int:
        col, score = self.minimax_prune(self.game.state, depth, -np.inf, np.inf, True)
        return col

    def get_best_move_3(self, depth: int):
        valid_moves = self.get_valid_moves()
        results = []

        for move in valid_moves:
            board_copy = copy(self.game.state)
            board_copy = board_copy.make_move(ConnectFourMove(move))
            results.append(self.minimax_prune(board_copy, depth, -np.inf, np.inf, True))

        print(results)
        max_results = [result[0] for result in results if result[1] == max([result[1] for result in results])]
        return choice(max_results)


    def minimax_prune(self, board, depth, alpha: float, beta: float,
                is_maximizing_player: bool) -> Tuple[int, float]:
        """Returns column index and score"""
        valid_moves = self.get_valid_moves()
        is_terminal = board.is_finished()

        if is_terminal or depth == 0:
            if is_terminal:
                if board.get_winner() == self.game.first_player:
                    return (None, 1e9)
                elif board.get_winner() == self.game.second_player:
                    return (None, -1e9)
                else:
                    return (None, 0)
            else:
                return (None, self.evaluate_position(board, self.game.second_player))

        if is_maximizing_player:
            value = -np.inf
            chosen_column = choice(valid_moves)
            for valid_move in valid_moves:
                board_copy = copy(board)
                board_copy = board_copy.make_move(ConnectFourMove(valid_move))

                evaluation = self.minimax_prune(board_copy, depth - 1, alpha, beta, False)[1]
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
                board_copy = copy(board)
                board_copy = board_copy.make_move(ConnectFourMove(valid_move))

                evaluation = self.minimax_prune(board_copy, depth - 1, alpha, beta, True)[1]

                if evaluation < value:
                    value = evaluation
                    chosen_column = valid_move
                beta = min(beta, value)

                if alpha >= beta:
                    break

            return chosen_column, value

    def run_evaluations(self, player):
        valid_moves = self.get_valid_moves()

        for move in valid_moves:
            board_copy = copy(self.game.state)
            board_copy = board_copy.make_move(ConnectFourMove(move))
            prize = self.evaluate_position(board_copy, player)
            print(f"Move: {move}\tPrize: {prize}")


"""if __name__ == "__main__":
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
"""

if __name__ == "__main__":
    p1 = Player("x")
    p2 = Player("o")
    DEPTH = 3
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1,
                       second_player=p2)

    algorithm = MinMaxSolver(game, ROW_COUNT, COLUMN_COUNT)
    player = p1

    i = 0
    #print(game)
    while not algorithm.game.state.is_finished():
        time.sleep(0.1)
        i += 1

        """chosen_column = input("Column: ")
        game.make_move(ConnectFourMove(int(chosen_column)))

        if algorithm.game.state.is_finished():
            break"""

        #print(f"Playing AI: {player.char}")
        best_move = algorithm.get_best_move_3(DEPTH)
        game.make_move(ConnectFourMove(best_move))

        print(game)





    print(f"Won: {game.get_winner().char}")
    print("Terminated")
    print(f"Moves: {i}")


"""if __name__ == "__main__":
    p1 = Player("a")
    p2 = Player("b")
    DEPTH = 3
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1,
                       second_player=p2)

    algorithm = MinMaxSolver(game, ROW_COUNT, COLUMN_COUNT)

    game.make_move(ConnectFourMove(3))
    print(game)
    print(algorithm.run_evaluations(p2))
    print(algorithm.get_best_move_2())
    print(game.state.get_current_player().char)"""
