
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

    raise ValueError("wasnt able to place all of the ships ")


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


class Game:
    def __init__(self, board_size, fleet, rng):
        self.board_size = board_size
        self.board = place_ships(board_size, fleet, rng)

        # 0 = unknown, 1 = miss, 2 = hit, 3 = sunk
        self.shots = np.zeros((board_size, board_size), dtype=np.int8)

        ship_lengths = np.bincount(self.board.ravel())[1:]
        self.cells_left = dict(enumerate(ship_lengths, start=1))
        self.ships_left = len(ship_lengths)

    @property
    def is_over(self):
        return self.ships_left == 0

    def shoot(self, row, col):
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("shot is outside board")

        if self.shots[row, col] != 0:
            raise ValueError("it have already shot at this cell")

        ship_id = self.board[row, col]

        if ship_id == 0:
            self.shots[row, col] = 1
            return "miss"

        self.cells_left[ship_id] -= 1

        if self.cells_left[ship_id] == 0:
            self.shots[self.board == ship_id] = 3
            self.ships_left -= 1
            return "sunk"

        self.shots[row, col] = 2
        return "hit"