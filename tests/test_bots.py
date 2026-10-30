import numpy as np
import pytest

from battleship.game import Game, SMALL_FLEET
from battleship.bots import RandomBot


def test_random_bot_finishes_a_game_without_repeating_shots():
    rng = np.random.default_rng(0)
    game = Game(6, SMALL_FLEET, rng)
    bot = RandomBot()
    shots_taken = 0

    while not game.is_over:
        row, col = bot.choose_shot(game.shots, rng)
        game.shoot(row, col)  # raises if the bot repeats a shot
        shots_taken += 1

    assert shots_taken <= 36


def test_random_bot_returns_plain_ints_on_unshot_cells():
    rng = np.random.default_rng(1)
    shots = np.ones((6, 6), dtype=np.int8)
    shots[2, 3] = 0  # the only cell left

    row, col = RandomBot().choose_shot(shots, rng)

    assert (row, col) == (2, 3)
    assert type(row) is int and type(col) is int


def test_random_bot_raises_when_no_cells_left():
    shots = np.ones((6, 6), dtype=np.int8)

    with pytest.raises(ValueError, match="No unshot cells"):
        RandomBot().choose_shot(shots, np.random.default_rng(0))