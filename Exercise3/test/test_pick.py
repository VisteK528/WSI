import unittest

from two_player_games.games.Pick import Pick, PickMove


class TestPick(unittest.TestCase):
    def test_init(self):
        game = Pick(n=3)
        self.assertEqual(str(game), "n: 3, aim_value: 15\nCurrent player: 1, Numbers: [],\nOther player: 2, Numbers: []")
        self.assertIs(game.get_current_player(), game.first_player)

        game = Pick(n=4)
        self.assertEqual(str(game), "n: 4, aim_value: 34\nCurrent player: 1, Numbers: [],\nOther player: 2, Numbers: []")
        self.assertIs(game.get_current_player(), game.first_player)

    def test_moves(self):
        game = Pick(n=3)

        game.make_move(PickMove(1))
        self.assertEqual(str(game), "n: 3, aim_value: 15\nCurrent player: 2, Numbers: [],\nOther player: 1, Numbers: [1]")
        self.assertIs(game.get_current_player(), game.second_player)

        game.make_move(PickMove(2))
        self.assertEqual(str(game), "n: 3, aim_value: 15\nCurrent player: 1, Numbers: [1],\nOther player: 2, Numbers: [2]")
        self.assertIs(game.get_current_player(), game.first_player)

        game.make_move(PickMove(5))
        game.make_move(PickMove(7))

        self.assertEqual(str(game), "n: 3, aim_value: 15\nCurrent player: 1, Numbers: [1, 5],\nOther player: 2, Numbers: [2, 7]")
        self.assertIs(game.get_current_player(), game.first_player)

    def test_get_moves(self):
        game = Pick(n=3)

        game.make_move(PickMove(5))
        game.make_move(PickMove(7))
        game.make_move(PickMove(3))

        self.assertSequenceEqual(
            game.get_moves(), [PickMove(number) for number in [1, 2, 4, 6, 8, 9]]
        )

        game = Pick(n=4)

        game.make_move(PickMove(5))
        game.make_move(PickMove(7))
        game.make_move(PickMove(3))
        game.make_move(PickMove(16))
        game.make_move(PickMove(15))
        game.make_move(PickMove(14))
        game.make_move(PickMove(12))

        self.assertSequenceEqual(
            game.get_moves(), [PickMove(number) for number in [1, 2, 4, 6, 8, 9, 10, 11, 13]]
        )

    def test_invalid_move(self):
        game = Pick(n=3)

        self.assertRaises(ValueError, lambda: game.make_move(PickMove(11)))

        game.make_move(PickMove(4))
        self.assertRaises(ValueError, lambda: game.make_move(PickMove(4)))

        game = Pick(n=4)

        self.assertRaises(ValueError, lambda: game.make_move(PickMove(21)))

        game.make_move(PickMove(4))
        self.assertRaises(ValueError, lambda: game.make_move(PickMove(4)))

    def test_winner_and_finished(self):
        game = Pick(n=3)

        game.make_move(PickMove(4))
        game.make_move(PickMove(9))
        game.make_move(PickMove(5))
        game.make_move(PickMove(6))

        self.assertFalse(game.is_finished())
        self.assertIsNone(game.get_winner())

        game.make_move(PickMove(3))
        game.make_move(PickMove(8))
        game.make_move(PickMove(7))

        self.assertTrue(game.is_finished())
        self.assertIs(game.get_winner(), game.first_player)

    def test_winner_and_finished_n4(self):
        game = Pick(n=4)

        game.make_move(PickMove(7))
        game.make_move(PickMove(1))
        game.make_move(PickMove(9))
        game.make_move(PickMove(2))

        self.assertFalse(game.is_finished())
        self.assertIsNone(game.get_winner())

        game.make_move(PickMove(12))
        game.make_move(PickMove(8))
        game.make_move(PickMove(6))

        self.assertTrue(game.is_finished())
        self.assertIs(game.get_winner(), game.first_player)

    def test_winner_and_finished_draw(self):
        game = Pick(n=3)

        game.make_move(PickMove(5))
        game.make_move(PickMove(8))
        game.make_move(PickMove(3))
        game.make_move(PickMove(7))

        self.assertFalse(game.is_finished())
        self.assertIsNone(game.get_winner())

        game.make_move(PickMove(9))
        game.make_move(PickMove(1))
        game.make_move(PickMove(6))
        game.make_move(PickMove(4))
        game.make_move(PickMove(2))

        self.assertEqual([], game.get_moves())
        self.assertTrue(game.is_finished())
        self.assertIsNone(game.get_winner())

