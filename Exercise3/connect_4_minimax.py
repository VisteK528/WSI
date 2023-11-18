from connect4_solvers import MinMaxSolver, HeuristicSolver, RandomSolver
from two_player_games.player import Player
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove
import time
from random import choice
ROW_COUNT = 5
COLUMN_COUNT = 7

if __name__ == "__main__":
    p1 = Player("x")
    p2 = Player("o")

    # Dobre głębokości przeszukiwań
    # 2, 4, 6, 8, 10
    DEPTH = 2

    p1_won = 0
    p2_won = 0
    games = 1
    for game_nb in range(games):
        game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1,
                           second_player=p2)

        algorithm = MinMaxSolver(game, ROW_COUNT, COLUMN_COUNT, max_player=p1, min_player=p2)

        algorithm2 = MinMaxSolver(game, ROW_COUNT, COLUMN_COUNT, max_player=p2, min_player=p1)

        heuristic_sovler = HeuristicSolver(game, ROW_COUNT, COLUMN_COUNT, max_player=p1, min_player=p2)

        i = 0
        while not algorithm.game.state.is_finished():
            time.sleep(0.1)

            i += 1
            # MiniMax alpha-beta
            #best_move = algorithm.get_best_move(DEPTH)
            #best_move = heuristic_sovler.get_best_move()
            best_move = int(input("Column: "))

            # Random move
            #best_move = choice(algorithm.get_valid_moves())

            # Heuristic move
            game.make_move(ConnectFourMove(best_move))
            print(game)

            if algorithm.game.state.is_finished():
                break

            i += 1
            best_move2 = algorithm2.get_best_move(DEPTH)
            # print(best_move2)
            game.make_move(ConnectFourMove(best_move2))
            print(game)

        print(game)
        print(f"Game: {game_nb}/{games}")
        print(f"Moves: {i}")

        if game.get_winner() is not None:
            if game.get_winner().char == p1.char:
                print(f"Won: {game.get_winner().char}")
                p1_won += 1
            elif game.get_winner().char == p2.char:
                print(f"Won: {game.get_winner().char}")
                p2_won += 1

    draws = games-p1_won-p2_won
    print("Stats")
    print(f"Games: {games}\t Draws: {draws}\tDraws percent: {round((draws/games)*100, 2)}")
    print(f"P1 winnings stats\t Won: {p1_won}\tLoosed: {p2_won} \tWin percent: {round((p1_won/games)*100, 2)}")
    print(
        f"P1 winnings stats\t Won: {p2_won}\tLoosed: {p1_won} \tWin percent: {round((p2_won / games) * 100, 2)}")




