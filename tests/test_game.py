import numpy as np
from battleship.game import place_ships, SMALL_FLEET, FULL_FLEET


def test_fleet_sizes():
    rng = np.random.default_rng(0)
    board = place_ships(6, SMALL_FLEET, rng)

    ship_counts = np.bincount(board.ravel())[1:]
    actual_lengths = sorted(ship_counts, reverse=True)

    assert actual_lengths == sorted(SMALL_FLEET, reverse=True)


def test_ships_dont_touch():
    rng = np.random.default_rng(1)

    for _ in range(2000):
        board = place_ships(6, SMALL_FLEET, rng)

        for row in range(6):
            for col in range(6):
                ship_id = board[row, col]

                if ship_id == 0:
                    continue

                # Check all neighbouring cells and diagonals.
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

    total_ship_cells = np.count_nonzero(board)

    assert total_ship_cells == sum(FULL_FLEET)


def test_same_seed_same_board():
    first_board = place_ships(6, SMALL_FLEET, np.random.default_rng(5))
    second_board = place_ships(6, SMALL_FLEET, np.random.default_rng(5))

    assert np.array_equal(first_board, second_board)


def test_ships_are_straight():
    rng = np.random.default_rng(3)

    for _ in range(500):
        board = place_ships(6, SMALL_FLEET, rng)

        for ship_id in range(1, len(SMALL_FLEET) + 1):
            ship_cells = np.argwhere(board == ship_id)

            rows = ship_cells[:, 0]
            cols = ship_cells[:, 1]

            # Every ship must occupy a single row or a single column.
            straight_line = len(set(rows)) == 1 or len(set(cols)) == 1
            assert straight_line

            # The rectangle must contain exactly the ship cells.
            height = rows.max() - rows.min() + 1
            width = cols.max() - cols.min() + 1

            assert height * width == len(ship_cells)