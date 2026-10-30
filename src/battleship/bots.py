import numpy as np


class RandomBot:
    #it shoots at random cell which it hasnt shot at yet

    def choose_shot(self, shots, rng):
        unshot_cells = np.argwhere(shots == 0)

        if len(unshot_cells) == 0:
            raise ValueError("no unshot cells left to shoot at")

        row, col = unshot_cells[rng.integers(len(unshot_cells))]

        return int(row), int(col)