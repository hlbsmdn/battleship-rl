
import numpy as np

from battleship.game import Game, SMALL_FLEET
from battleship.bots import RandomBot

# i fixed test boards so results are reproducible.
# i have to keep these seeds separate from those used during development
TEST_SEED_START = 1_000_000


def play_game(bot, board_size, fleet, board_seed, bot_seed):
    game = Game(board_size, fleet, np.random.default_rng(board_seed))
    bot_rng = np.random.default_rng(bot_seed)
    shots_taken = 0

    while not game.is_over:
        row, col = bot.choose_shot(game.shots, bot_rng)
        game.shoot(row, col)
        shots_taken += 1

    return shots_taken


def evaluate(bot, board_size, fleet, n_boards=1000):
    if n_boards <= 0:
        raise ValueError("n_boards must be positive")

    results = [
        play_game(
            bot,
            board_size,
            fleet,
            TEST_SEED_START + i,
            i
        )
        for i in range(n_boards)
    ]

    return np.array(results)


if __name__ == "__main__":
    results = evaluate(RandomBot(), 6, SMALL_FLEET)

    print(
        f"random bot: mean {results.mean():.1f}, "
        f"median {np.median(results):.0f}, "
        f"std {results.std():.1f}, "
        f"best {results.min()}, worst {results.max()}"
    )
