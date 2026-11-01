import numpy as np
import pytest

from battleship.replay import ReplayBuffer


def add_transition(buffer, action, done=False):
    observation = np.full((4, 6, 6), action, dtype=np.float32)
    next_observation = np.full((4, 6, 6), action + 1, dtype=np.float32)
    legal = np.ones(36, dtype=bool)

    buffer.add(
        observation,
        action,
        float(action),
        next_observation,
        done,
        legal,
    )


def test_add_increases_length():
    buffer = ReplayBuffer(100, 6)

    assert len(buffer) == 0

    add_transition(buffer, 0)
    add_transition(buffer, 1)

    assert len(buffer) == 2


def test_sample_returns_correct_shapes():
    buffer = ReplayBuffer(100, 6)

    for action in range(20):
        add_transition(buffer, action)

    observations, actions, rewards, next_observations, dones, next_legal = (
        buffer.sample(8, np.random.default_rng(0))
    )

    assert observations.shape == (8, 4, 6, 6)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_observations.shape == (8, 4, 6, 6)
    assert dones.shape == (8,)
    assert next_legal.shape == (8, 36)


def test_sample_keeps_transition_data_together():
    buffer = ReplayBuffer(100, 6)

    for action in range(20):
        add_transition(buffer, action)

    observations, actions, rewards, next_observations, _, _ = (
        buffer.sample(10, np.random.default_rng(0))
    )

    assert np.array_equal(rewards, actions)
    assert np.array_equal(observations[:, 0, 0, 0], actions)
    assert np.array_equal(next_observations[:, 0, 0, 0], actions + 1)


def test_full_buffer_overwrites_oldest():
    buffer = ReplayBuffer(3, 6)

    for action in range(5):
        add_transition(buffer, action)

    assert len(buffer) == 3
    assert sorted(buffer.actions.tolist()) == [2, 3, 4]


def test_sample_does_not_repeat_transitions():
    buffer = ReplayBuffer(100, 6)

    for action in range(20):
        add_transition(buffer, action)

    _, actions, _, _, _, _ = buffer.sample(
        10, np.random.default_rng(0)
    )

    assert len(set(actions.tolist())) == 10


def test_sample_raises_when_batch_is_too_large():
    buffer = ReplayBuffer(100, 6)
    add_transition(buffer, 0)

    with pytest.raises(ValueError, match="not enough"):
        buffer.sample(2, np.random.default_rng(0))


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(0, 6)


def test_board_size_must_be_positive():
    with pytest.raises(ValueError, match="board_size"):
        ReplayBuffer(100, 0)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size):
    buffer = ReplayBuffer(10, 6)
    add_transition(buffer, 0)

    with pytest.raises(ValueError, match="batch_size"):
        buffer.sample(batch_size, np.random.default_rng(0))


def test_done_and_next_legal_are_stored():
    buffer = ReplayBuffer(10, 6)

    legal = np.ones(36, dtype=bool)
    legal[[3, 7]] = False

    buffer.add(
        np.zeros((4, 6, 6), dtype=np.float32),
        5,
        -1.0,
        np.zeros((4, 6, 6), dtype=np.float32),
        True,
        legal,
    )

    _, _, _, _, dones, next_legal = buffer.sample(1, np.random.default_rng(0))

    assert dones[0] == 1.0
    assert np.array_equal(next_legal[0], legal)