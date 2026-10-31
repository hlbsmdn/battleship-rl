
import numpy as np
import pytest

from battleship.env import BattleshipEnv
from battleship.game import SMALL_FLEET


def make_env(seed=0):
    return BattleshipEnv(6, SMALL_FLEET, seed)


def find_action_for_result(seed, channel):
    for action in range(36):
        env = make_env(seed)
        env.reset()

        observation, _, _ = env.step(action)
        row, col = divmod(action, 6)

        if observation[channel, row, col] == 1:
            return action

    raise AssertionError("no action found which matches")


def test_reset_returns_correct_observation_shape():
    env = make_env()

    observation = env.reset()

    assert observation.shape == (4, 6, 6)
    assert observation.dtype == np.float32


def test_observation_only_contains_zero_and_one():
    env = make_env()
    observation = env.reset()

    assert np.isin(observation, [0.0, 1.0]).all()


def test_initial_observation_contains_no_shots():
    env = make_env()
    observation = env.reset()

    assert observation[0].all()
    assert observation[1:].sum() == 0


def test_legal_actions_only_include_unshot_cells():
    env = make_env()
    env.reset()

    legal = env.legal_actions()

    assert legal.shape == (36,)
    assert legal.all()


def test_step_returns_expected_values():
    env = make_env()
    env.reset()

    action = int(np.flatnonzero(env.legal_actions())[0])

    observation, reward, done = env.step(action)

    assert observation.shape == (4, 6, 6)
    assert reward == -1.0
    assert done is False


def test_step_updates_legal_actions():
    env = make_env()
    env.reset()

    action = 10
    env.step(action)

    assert not env.legal_actions()[action]
    assert env.legal_actions().sum() == 35


def test_step_updates_observation_after_miss():
    env = make_env(seed=1)
    env.reset()

    action = find_action_for_result(1, 1)
    row, col = divmod(action, 6)

    observation, _, _ = env.step(action)

    assert observation[0, row, col] == 0
    assert observation[1, row, col] == 1
    assert observation[2, row, col] == 0
    assert observation[3, row, col] == 0


def test_step_updates_observation_after_hit():
    env = make_env(seed=1)
    env.reset()

    action = find_action_for_result(1, 2)
    row, col = divmod(action, 6)

    observation, _, _ = env.step(action)

    assert observation[0, row, col] == 0
    assert observation[1, row, col] == 0
    assert observation[2, row, col] == 1
    assert observation[3, row, col] == 0


def test_step_rejects_actions_outside_board():
    env = make_env()
    env.reset()

    with pytest.raises(ValueError, match="outside the board"):
        env.step(-1)

    with pytest.raises(ValueError, match="outside the board"):
        env.step(36)


def test_step_rejects_repeated_actions():
    env = make_env()
    env.reset()

    env.step(10)

    with pytest.raises(ValueError, match="already been taken"):
        env.step(10)


def test_step_requires_reset():
    env = make_env()

    with pytest.raises(ValueError, match="call reset"):
        env.step(0)


def test_step_rejects_actions_after_game_over():
    env = make_env()
    env.reset()

    done = False

    while not done:
        action = int(np.flatnonzero(env.legal_actions())[0])
        _, _, done = env.step(action)

    with pytest.raises(ValueError, match="game is over"):
        env.step(0)


def test_reset_starts_new_game():
    env = make_env()
    env.reset()

    env.step(0)
    observation = env.reset()

    assert observation.shape == (4, 6, 6)
    assert env.legal_actions().all()


def test_same_seed_produces_same_sequence():
    first = make_env(seed=42)
    second = make_env(seed=42)

    first_observation = first.reset()
    second_observation = second.reset()

    assert np.array_equal(first_observation, second_observation)

    for _ in range(10):
        first_action = int(np.flatnonzero(first.legal_actions())[0])
        second_action = int(np.flatnonzero(second.legal_actions())[0])

        first_result = first.step(first_action)
        second_result = second.step(second_action)

        assert np.array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]


def test_game_can_be_completed():
    env = make_env()
    env.reset()

    steps = 0
    done = False

    while not done:
        legal_actions = np.flatnonzero(env.legal_actions())
        action = int(legal_actions[0])

        _, reward, done = env.step(action)

        assert reward == -1.0
        steps += 1

    assert done
    assert steps <= 36


def test_reward_is_same_for_every_shot():
    env = make_env()
    env.reset()

    rewards = []

    for action in np.flatnonzero(env.legal_actions())[:10]:
        _, reward, _ = env.step(int(action))
        rewards.append(reward)

    assert all(reward == -1.0 for reward in rewards)


def test_each_cell_is_in_exactly_one_layer():
    env = make_env()
    env.reset()
    env.step(0)
    observation, _, _ = env.step(5)

    assert (observation.sum(axis=0) == 1).all()


def test_sunk_ship_shows_in_sunk_layer():
    env = make_env()
    env.reset()

    row, col = np.argwhere(env.game.board > 0)[0]
    ship_id = env.game.board[row, col]
    ship_cells = np.argwhere(env.game.board == ship_id)

    for row, col in ship_cells:
        observation, _, _ = env.step(row * 6 + col)

    for row, col in ship_cells:
        assert observation[3, row, col] == 1
        assert observation[2, row, col] == 0