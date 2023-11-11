import unittest

from two_player_games.games.morris import SixMensMorris, MorrisMove


class TestMorris(unittest.TestCase):
    def test_init(self):
        game = SixMensMorris()
        self.assertSequenceEqual(game.state.grid, [None] * 16)
        self.assertDictEqual(game.state.placed_pawns, {game.first_player: 0, game.second_player: 0})
        self.assertEqual(game.state.n_moves, 0)

    def test_moves(self):
        game = SixMensMorris()
        p1, p2 = game.get_players()

        game.make_move(MorrisMove(None, 0, None))
        game.make_move(MorrisMove(None, 2, None))
        game.make_move(MorrisMove(None, 5, None))
        game.make_move(MorrisMove(None, 13, None))
        game.make_move(MorrisMove(None, 12, None))
        game.make_move(MorrisMove(None, 8, None))
        game.make_move(MorrisMove(None, 7, None))
        game.make_move(MorrisMove(None, 3, None))
        game.make_move(MorrisMove(None, 11, None))
        game.make_move(MorrisMove(None, 4, 12))
        game.make_move(MorrisMove(None, 14, None))
        game.make_move(MorrisMove(None, 9, None))

        self.assertDictEqual(game.state.placed_pawns, {game.first_player: 6, game.second_player: 6})
        self.assertSequenceEqual(
            game.state.grid,
            [p1, None, p2, p2, p2, p1, None, p1, p2, p2, None, p1, None, p2, p1, None]
        )
        self.assertEqual(game.state.n_moves, 2)

        game.make_move(MorrisMove(14, 15, None))
        game.make_move(MorrisMove(9, 10, None))
        game.make_move(MorrisMove(5, 6, 8))

        self.assertSequenceEqual(
            game.state.grid,
            [p1, None, p2, p2, p2, None, p1, p1, None, None, p2, p1, None, p2, None, p1]
        )
        self.assertEqual(game.state.n_moves, 0)

    def test_get_moves(self):
        game = SixMensMorris()

        game.make_move(MorrisMove(None, 0, None))
        game.make_move(MorrisMove(None, 2, None))
        game.make_move(MorrisMove(None, 5, None))
        game.make_move(MorrisMove(None, 13, None))
        game.make_move(MorrisMove(None, 12, None))
        game.make_move(MorrisMove(None, 8, None))

        self.assertSequenceEqual(
            game.get_moves(),
            [MorrisMove(None, i, None) for i in [1, 3, 4, 6, 7, 9, 10, 11, 14, 15]]
        )

        game.make_move(MorrisMove(None, 7, None))
        game.make_move(MorrisMove(None, 3, None))
        game.make_move(MorrisMove(None, 11, None))

        self.assertSequenceEqual(
            game.get_moves(),
            [
                MorrisMove(None, 1, None),
                *[MorrisMove(None, 4, i) for i in [0, 5, 7, 11, 12]],
                *[MorrisMove(None, i, None) for i in [6, 9, 10, 14, 15]]
            ]
        )

        game.make_move(MorrisMove(None, 4, 12))

        self.assertSequenceEqual(
            game.get_moves(),
            [
                MorrisMove(None, 1, None),
                *[MorrisMove(None, 6, i) for i in [8, 13]],  # 2, 3, 4 make a morris
                *[MorrisMove(None, i, None) for i in [9, 10, 12, 14, 15]]
            ]
        )

        game.make_move(MorrisMove(None, 14, None))
        game.make_move(MorrisMove(None, 9, None))

        game.make_move(MorrisMove(14, 15, None))
        game.make_move(MorrisMove(9, 10, None))
        game.make_move(MorrisMove(5, 6, 8))

        self.assertSequenceEqual(
            game.get_moves(),
            [
                MorrisMove(2, 1, None), MorrisMove(4, 5, None), MorrisMove(10, 9, None),
                MorrisMove(13, 12, None), MorrisMove(13, 14, None), MorrisMove(13, 5, None),
            ]
        )

        game.make_move(MorrisMove(4, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(5, 4, 11))
        game.make_move(MorrisMove(1, 0, 10))
        game.make_move(MorrisMove(4, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(13, 12, None))
        game.make_move(MorrisMove(1, 0, 12))
        game.make_move(MorrisMove(5, 11, None))
        game.make_move(MorrisMove(15, 8, None))
        game.make_move(MorrisMove(11, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(5, 4, 8))

        self.assertSequenceEqual(
            game.get_moves(),
            [
                *[MorrisMove(1, 0, i) for i in [2, 3, 4]],  # 2, 3, 4 make a morris, but only those are left
                MorrisMove(6, 5, None), MorrisMove(7, 0, None), MorrisMove(1, 9, None), MorrisMove(7, 15, None),
            ]
        )

    def test_winner_and_finished_empty(self):
        game = SixMensMorris()
        self.assertFalse(game.is_finished())

    def test_winner_and_finished(self):
        game = SixMensMorris()
        p1, p2 = game.get_players()

        game.make_move(MorrisMove(None, 0, None))
        game.make_move(MorrisMove(None, 2, None))
        game.make_move(MorrisMove(None, 5, None))
        game.make_move(MorrisMove(None, 13, None))
        game.make_move(MorrisMove(None, 12, None))
        game.make_move(MorrisMove(None, 8, None))
        game.make_move(MorrisMove(None, 7, None))
        game.make_move(MorrisMove(None, 3, None))
        game.make_move(MorrisMove(None, 11, None))
        game.make_move(MorrisMove(None, 4, 12))
        game.make_move(MorrisMove(None, 14, None))
        game.make_move(MorrisMove(None, 9, None))
        game.make_move(MorrisMove(14, 15, None))
        game.make_move(MorrisMove(9, 10, None))
        game.make_move(MorrisMove(5, 6, 8))
        game.make_move(MorrisMove(4, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(5, 4, 11))
        game.make_move(MorrisMove(1, 0, 10))
        game.make_move(MorrisMove(4, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(13, 12, None))
        game.make_move(MorrisMove(1, 0, 12))
        game.make_move(MorrisMove(5, 11, None))
        game.make_move(MorrisMove(15, 8, None))
        game.make_move(MorrisMove(11, 5, None))
        game.make_move(MorrisMove(0, 1, None))
        game.make_move(MorrisMove(5, 4, 8))

        self.assertFalse(game.is_finished())
        game.make_move(MorrisMove(1, 0, 2))
        self.assertTrue(game.is_finished())
        self.assertIs(game.get_winner(), p1)

    def test_winner_and_finished_no_possible_moves(self):
        game = SixMensMorris()
        p1, p2 = game.get_players()

        game.make_move(MorrisMove(None, 0, None))
        game.make_move(MorrisMove(None, 7, None))
        game.make_move(MorrisMove(None, 1, None))
        game.make_move(MorrisMove(None, 8, None))
        game.make_move(MorrisMove(None, 3, None))
        game.make_move(MorrisMove(None, 2, None))
        game.make_move(MorrisMove(None, 4, None))
        game.make_move(MorrisMove(None, 11, None))
        game.make_move(MorrisMove(None, 9, None))
        self.assertFalse(game.is_finished())
        game.make_move(MorrisMove(None, 6, None))
        game.make_move(MorrisMove(None, 10, None))
        self.assertFalse(game.is_finished())
        game.make_move(MorrisMove(6, 5, None))
        self.assertTrue(game.is_finished())
        self.assertIs(game.get_winner(), p2)
