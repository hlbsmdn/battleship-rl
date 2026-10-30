import numpy as np
import pytest

from battleship.game import place_ships, Game, SMALL_FLEET, FULL_FLEET


def make_game(seed=0):
    return Game(6, SMALL_FLEET, np.random.default_rng(seed))


def test_fleet_sizes():
    rng = np.random.default_rng(0)
    board = place_ships(6, SMALL_FLEET, rng)

    ship_counts = np.bincount(board.ravel())[1:]
    actual_lengths = sorted(ship_counts, reverse=True)

    assert actual_lengths == sorted(SMALL_FLEET, reverse=True)


def test_ships_dont_touch():
    rng = np.random.default_rng(1)

    for _ in range(500):
        board = place_ships(6, SMALL_FLEET, rng)

        for row in range(6):
            for col in range(6):
                ship_id = board[row, col]

                if ship_id == 0:
                    continue

                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        neighbour_row = row + row_offset
                        neighbour_col = col + col_offset

                        if 0 <= neighbour_row < 6 and 0 <= neighbour_col < 6:
                            neighbour_id = board[neighbour_row, neighbour_col]
                            assert neighbour_id == 0 or neighbour_id == ship_id


def test_full_fleet_fits():
    rng = np.random.default_rng(2)
    board = place_ships(10, FULL_FLEET, rng)

    assert np.count_nonzero(board) == sum(FULL_FLEET)


def test_same_seed_same_board():
    first_board = place_ships(6, SMALL_FLEET, np.random.default_rng(5))
    second_board = place_ships(6, SMALL_FLEET, np.random.default_rng(5))

    assert np.array_equal(first_board, second_board)


def test_ships_are_straight_and_continuous():
    rng = np.random.default_rng(3)

    for _ in range(200):
        board = place_ships(6, SMALL_FLEET, rng)

        for ship_id in range(1, len(SMALL_FLEET) + 1):
            ship_cells = np.argwhere(board == ship_id)

            rows = ship_cells[:, 0]
            cols = ship_cells[:, 1]

            assert len(set(rows)) == 1 or len(set(cols)) == 1

            height = rows.max() - rows.min() + 1
            width = cols.max() - cols.min() + 1

            assert height * width == len(ship_cells)


def test_miss():
    game = make_game()
    row, col = np.argwhere(game.board == 0)[0]

    assert game.shoot(row, col) == "miss"
    assert game.shots[row, col] == 1


def test_hit_then_sunk():
    game = make_game()

    row, col = np.argwhere(game.board > 0)[0]
    ship_id = game.board[row, col]
    ship_cells = np.argwhere(game.board == ship_id)

    for row, col in ship_cells[:-1]:
        assert game.shoot(row, col) == "hit"

    row, col = ship_cells[-1]
    assert game.shoot(row, col) == "sunk"
    assert (game.shots[game.board == ship_id] == 3).all()


def test_repeat_shot_raises():
    game = make_game()
    game.shoot(0, 0)

    with pytest.raises(ValueError, match="it have already shot at this cell"):
        game.shoot(0, 0)


def test_shot_outside_board_raises():
    game = make_game()

    with pytest.raises(ValueError, match="shot is outside board"):
        game.shoot(-1, 0)

    with pytest.raises(ValueError, match="shot is outside board"):
        game.shoot(0, 6)


def test_game_over_after_sinking_everything():
    game = make_game()
    assert not game.is_over

    for row, col in np.argwhere(game.board > 0):
        if game.shots[row, col] == 0:
            game.shoot(row, col)

    assert game.is_over


def test_unshot_ships_stay_hidden():
    game = make_game()

    row, col = np.argwhere(game.board > 0)[0]
    game.shoot(row, col)

    hidden_ship_cells = (game.board > 0) & (game.shots == 0)
    assert hidden_ship_cells.sum() == sum(SMALL_FLEET) - 1