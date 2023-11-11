import unittest
from two_player_games.games.dots_and_boxes import DotsAndBoxes, DotsAndBoxesMove


class TestDotsAndBoxes(unittest.TestCase):
    def test_init(self):
        game = DotsAndBoxes(2)
        self.assertEqual(str(game), "Current player: 1\no o o\n     \no o o\n     \no o o")
        self.assertIs(game.get_current_player(), game.first_player)

    def test_moves(self):
        game = DotsAndBoxes(2)

        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        self.assertEqual(str(game), "Current player: 2\no o o\n     \no o-o\n     \no o o")
        self.assertIs(game.get_current_player(), game.second_player)

        game.make_move(DotsAndBoxesMove('v', (1, 0)))
        self.assertEqual(str(game), "Current player: 1\no o o\n     \no o-o\n|    \no o o")
        self.assertIs(game.get_current_player(), game.first_player)

        game.make_move(DotsAndBoxesMove('v', (1, 1)))
        game.make_move(DotsAndBoxesMove('h', (1, 2)))

        self.assertEqual(str(game), "Current player: 1\no o o\n     \no o-o\n| |  \no o-o")
        self.assertIs(game.get_current_player(), game.first_player)

        game.make_move(DotsAndBoxesMove('v', (1, 2)))

        self.assertEqual(str(game), "Current player: 1\no o o\n     \no o-o\n| |1|\no o-o")
        self.assertIs(game.get_current_player(), game.first_player)

    def test_get_moves(self):
        game = DotsAndBoxes(2)

        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        game.make_move(DotsAndBoxesMove('v', (1, 0)))

        game.make_move(DotsAndBoxesMove('v', (1, 1)))
        game.make_move(DotsAndBoxesMove('h', (1, 2)))

        self.assertSequenceEqual(
            game.get_moves(),
            (
                DotsAndBoxesMove('h', (0, 0)),
                DotsAndBoxesMove('h', (0, 1)),
                DotsAndBoxesMove('h', (0, 2)),
                DotsAndBoxesMove('h', (1, 0)),
                DotsAndBoxesMove('v', (0, 0)),
                DotsAndBoxesMove('v', (0, 1)),
                DotsAndBoxesMove('v', (0, 2)),
                DotsAndBoxesMove('v', (1, 2)),
            )
        )

    def test_invalid_move(self):
        game = DotsAndBoxes(2)
        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        self.assertRaises(ValueError, lambda: game.make_move(DotsAndBoxesMove('h', (1, 1))))

    def test_score(self):
        
        game = DotsAndBoxes(2)

        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        game.make_move(DotsAndBoxesMove('v', (1, 0)))
        game.make_move(DotsAndBoxesMove('v', (1, 1)))
        game.make_move(DotsAndBoxesMove('h', (1, 2)))
        game.make_move(DotsAndBoxesMove('v', (1, 2)))
        game.make_move(DotsAndBoxesMove('h', (0, 2)))
        game.make_move(DotsAndBoxesMove('h', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (0, 2)))
        game.make_move(DotsAndBoxesMove('h', (1, 0)))

        scores = game.state.get_scores()
        self.assertDictEqual(scores, {game.first_player: 1, game.second_player: 2})

    def test_winner_and_finished(self):
        game = DotsAndBoxes(2)

        game.make_move(DotsAndBoxesMove('v', (0, 0)))
        game.make_move(DotsAndBoxesMove('h', (0, 0)))
        game.make_move(DotsAndBoxesMove('v', (1, 0)))
        game.make_move(DotsAndBoxesMove('h', (1, 2)))
        game.make_move(DotsAndBoxesMove('h', (0, 2)))
        game.make_move(DotsAndBoxesMove('v', (1, 2)))
        game.make_move(DotsAndBoxesMove('v', (0, 2)))
        game.make_move(DotsAndBoxesMove('h', (1, 0)))

        self.assertFalse(game.is_finished())
        self.assertIsNone(game.get_winner())

        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        game.make_move(DotsAndBoxesMove('h', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (1, 1)))

        self.assertTrue(game.is_finished())
        self.assertIs(game.get_winner(), game.first_player)

    def test_winner_and_finished_draw(self):
        game = DotsAndBoxes(2)

        game.make_move(DotsAndBoxesMove('h', (1, 1)))
        game.make_move(DotsAndBoxesMove('v', (1, 0)))
        game.make_move(DotsAndBoxesMove('v', (1, 1)))
        game.make_move(DotsAndBoxesMove('h', (1, 2)))
        game.make_move(DotsAndBoxesMove('v', (1, 2)))
        game.make_move(DotsAndBoxesMove('h', (0, 2)))
        game.make_move(DotsAndBoxesMove('h', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (0, 1)))
        game.make_move(DotsAndBoxesMove('v', (0, 2)))
        game.make_move(DotsAndBoxesMove('h', (1, 0)))

        self.assertFalse(game.is_finished())
        self.assertIsNone(game.get_winner())

        game.make_move(DotsAndBoxesMove('v', (0, 0)))
        game.make_move(DotsAndBoxesMove('h', (0, 0)))

        self.assertTrue(game.is_finished())
        self.assertIsNone(game.get_winner())
