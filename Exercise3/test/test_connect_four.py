from two_player_games.player import Player
import unittest
from two_player_games.games.connect_four import ConnectFour, ConnectFourMove, ConnectFourState


class TestConnectFour(unittest.TestCase):
    def test_init(self):
        game = ConnectFour()
        self.assertEqual(len(game.state.fields), 7)
        self.assertEqual(len(game.state.fields[0]), 6)

    def test_make_move(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)
        game.make_move(ConnectFourMove(2))
        self.assertListEqual(game.state.fields[2], [p1, None, None, None, None, None])
        game.make_move(ConnectFourMove(2))
        self.assertListEqual(game.state.fields[2], [p1, p2, None, None, None, None])

    def test_get_moves(self):
        game = ConnectFour()
        self.assertSequenceEqual(game.get_moves(), [ConnectFourMove(i) for i in range(7)])
        for _ in range(6):
            game.make_move(ConnectFourMove(2))
        for _ in range(6):
            game.make_move(ConnectFourMove(5))
        for _ in range(3):
            game.make_move(ConnectFourMove(0))
        self.assertSequenceEqual(game.get_moves(), [ConnectFourMove(i) for i in range(7) if i not in (2, 5)])

    def test_check_winner_vertical(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        self.assertIsNone(game.get_winner())
        game.make_move(ConnectFourMove(1))
        self.assertIs(game.get_winner(), p1)

    def test_check_winner_horizontal(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(3))
        self.assertIsNone(game.get_winner())
        game.make_move(ConnectFourMove(4))
        self.assertIs(game.get_winner(), p1)

    def test_check_winner_diagonal1(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(6))
        self.assertIsNone(game.get_winner())
        game.make_move(ConnectFourMove(1))
        self.assertIs(game.get_winner(), p1)

    def test_check_winner_diagonal2(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(6))
        self.assertIsNone(game.get_winner())
        game.make_move(ConnectFourMove(4))
        self.assertIs(game.get_winner(), p1)

    def test_str(self):
        p1 = Player('a')
        p2 = Player('b')
        game = ConnectFour(first_player=p1, second_player=p2)

        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(1))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(2))
        game.make_move(ConnectFourMove(4))
        game.make_move(ConnectFourMove(3))
        game.make_move(ConnectFourMove(6))

        self.assertEqual(
            str(game), "Current player: a\n"
            + "[ ][ ][ ][ ][ ][ ][ ]\n"
            + "[ ][ ][ ][ ][ ][ ][ ]\n"
            + "[ ][ ][ ][ ][ ][ ][ ]\n"
            + "[ ][ ][ ][a][b][ ][ ]\n"
            + "[ ][ ][a][b][b][ ][ ]\n"
            + "[ ][a][a][b][a][ ][b]")

    def test_winner_final_row(self):
        p1 = Player('1')
        p2 = Player('2')

        state = ConnectFourState(fields=[
            [p1, p1, p1, p2, p1, p1],
            [p1, p2, p2, p1, p2, None],
            [p2, p1, p2, p2, None, None],
            [p2, p2, None, None, None, None],
            [p1, p2, p1, p2, p2, p1],
            [p2, p2, p2, p1, p1, p1],
            [p1, p2, p1, None, None, None]
        ], current_player=p1)

        self.assertIs(state.get_winner(), p2)

    def test_tie(self):
        p1 = Player('1')
        p2 = Player('2')

        state = ConnectFourState(fields=[
            [p1, p1, p1, p2, p1, p1],
            [p1, p2, p2, p1, p2, p2],
            [p2, p1, p2, p2, p1, p1],
            [p2, p2, p1, p1, p1, p2],
            [p1, p2, p1, p2, p2, p1],
            [p2, p2, p2, p1, p1, p1],
            [p2, p1, p1, p1, p2, p2]
        ], current_player=p1)

        self.assertIs(state.get_winner(), None)
        self.assertTrue(state.is_finished())