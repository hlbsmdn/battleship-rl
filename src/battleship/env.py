
import numpy as np

from battleship.game import Game

# have to jeep training seeds separate from evaluation seeds
TRAIN_SEED_LIMIT = 1_000_000

STEP_PENALTY = -1.0


class BattleshipEnv:
    def __init__(self, board_size, fleet, seed=None):
        self.board_size = board_size
        self.fleet = fleet
        self.rng = np.random.default_rng(seed)
        self.game = None

    def reset(self):
        board_seed = int(self.rng.integers(0, TRAIN_SEED_LIMIT))
        self.game = Game(
            self.board_size,
            self.fleet,
            np.random.default_rng(board_seed)
        )

        return self.observation()

    def observation(self):
        shots = self.game.shots

        return np.stack([
            shots == 0,
            shots == 1,
            shots == 2,
            shots == 3
        ]).astype(np.float32)

    def legal_actions(self):
        return (self.game.shots == 0).ravel()

    def step(self, action):
        if self.game is None:
            raise ValueError("call reset() before step()")

        if self.game.is_over:
            raise ValueError("game is over, call reset()")

        action = int(action)

        if not 0 <= action < self.board_size ** 2:
            raise ValueError("action is outside the board")

        if not self.legal_actions()[action]:
            raise ValueError("action has already been taken")

        row, col = divmod(action, self.board_size)
        self.game.shoot(row, col)

        return self.observation(), self._reward(), self.game.is_over

    def _reward(self):
        return STEP_PENALTY
