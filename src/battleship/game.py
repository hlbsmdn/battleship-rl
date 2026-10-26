
import numpy as np

SMALL_FLEET = [3, 2, 2, 1, 1]
FULL_FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]


def place_ships(board_size, fleet, rng, max_attempts=1000):
    ships = sorted(fleet, reverse=True)

    for _ in range(max_attempts):
        board = np.zeros((board_size, board_size), dtype=np.int8)
        success = True

        for ship_id, length in enumerate(ships, start=1):
            placed = _place_one(board, length, ship_id, rng)

            if not placed:
                success = False
                break

        if success:
            return board

    raise ValueError("Could not place all ships")


def _place_one(board, length, ship_id, rng, max_tries=200):
    size = board.shape[0]

    for _ in range(max_tries):
        if rng.random() < 0.5:
            height = 1
            width = length
        else:
            height = length
            width = 1

        row = rng.integers(0, size - height + 1)
        col = rng.integers(0, size - width + 1)

        top = max(0, row - 1)
        bottom = min(size, row + height + 1)
        left = max(0, col - 1)
        right = min(size, col + width + 1)

        area = board[top:bottom, left:right]

        if np.any(area != 0):
            continue

        board[row:row + height, col:col + width] = ship_id
        return True

    return False


def show(board):
    for row in board:
        for cell in row:
            if cell == 0:
                print(".", end=" ")
            else:
                print("#", end=" ")
        print()
