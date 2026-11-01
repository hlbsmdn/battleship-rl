import copy

import numpy as np
import pytest
import torch
from torch import nn

from battleship.agent import (
    QNetwork,
    compute_targets,
    dqn_update,
    sync_target,
)


class FixedNetwork(nn.Module):
    def __init__(self, q_values):
        super().__init__()
        self.register_buffer(
            "q_values",
            torch.tensor([q_values], dtype=torch.float32),
        )

    def forward(self, observations):
        return self.q_values.expand(observations.shape[0], -1)


def target_for(online, target, rewards, dones, legal, gamma=0.9):
    observations = torch.zeros(len(rewards), 4, 1, 1)

    return compute_targets(
        online,
        target,
        torch.tensor(rewards, dtype=torch.float32),
        observations,
        torch.tensor(dones, dtype=torch.float32),
        torch.tensor(legal, dtype=torch.bool),
        gamma,
    )


def test_terminal_state_returns_reward():
    net = FixedNetwork([1, 2, 3])

    result = target_for(
        net,
        net,
        rewards=[-1],
        dones=[1],
        legal=[[True, True, True]],
    )

    assert torch.equal(result, torch.tensor([-1.0]))


def test_online_network_selects_action():
    online = FixedNetwork([0, 10, 0])
    target = FixedNetwork([100, 5, 50])

    result = target_for(
        online,
        target,
        rewards=[-1],
        dones=[0],
        legal=[[True, True, True]],
    )

    # The online network picks action 1; the target value is 5.
    assert torch.allclose(result, torch.tensor([3.5]))


def test_illegal_action_is_excluded():
    online = FixedNetwork([100, 1, 10])
    target = FixedNetwork([100, 5, 50])

    result = target_for(
        online,
        target,
        rewards=[-1],
        dones=[0],
        legal=[[False, True, True]],
    )

    assert torch.allclose(result, torch.tensor([44.0]))


def test_nonterminal_state_needs_a_legal_action():
    net = FixedNetwork([1, 2, 3])

    with pytest.raises(ValueError, match="legal action"):
        target_for(
            net,
            net,
            rewards=[-1],
            dones=[0],
            legal=[[False, False, False]],
        )


def test_terminal_rows_do_not_bootstrap():
    online = FixedNetwork([0, 10, 0])
    target = FixedNetwork([100, 5, 50])

    result = target_for(
        online,
        target,
        rewards=[-1, -1],
        dones=[1, 0],
        legal=[
            [False, False, False],
            [True, True, True],
        ],
    )

    assert torch.allclose(result, torch.tensor([-1.0, 3.5]))


def test_all_terminal_batch_returns_rewards():
    net = FixedNetwork([1, 2, 3])
    rewards = torch.tensor([-1.0, -2.0])

    result = target_for(
        net,
        net,
        rewards=rewards.tolist(),
        dones=[1, 1],
        legal=[
            [False, False, False],
            [False, False, False],
        ],
    )

    assert torch.equal(result, rewards)


def make_batch():
    rng = np.random.default_rng(0)

    return (
        rng.random((16, 4, 6, 6), dtype=np.float32),
        rng.integers(0, 36, size=16, dtype=np.int64),
        np.full(16, -1.0, dtype=np.float32),
        rng.random((16, 4, 6, 6), dtype=np.float32),
        np.ones(16, dtype=np.float32),
        np.ones((16, 36), dtype=np.bool_),
    )


def test_update_changes_online_weights_only():
    torch.manual_seed(0)

    online = QNetwork()
    target = copy.deepcopy(online)
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

    online_before = [
        parameter.detach().clone()
        for parameter in online.parameters()
    ]
    target_before = [
        parameter.detach().clone()
        for parameter in target.parameters()
    ]

    loss = dqn_update(
        online,
        target,
        optimizer,
        make_batch(),
        gamma=0.99,
    )

    assert np.isfinite(loss)

    assert any(
        not torch.equal(old, new)
        for old, new in zip(online_before, online.parameters())
    )

    assert all(
        torch.equal(old, new)
        for old, new in zip(target_before, target.parameters())
    )


def test_sync_target_copies_state():
    online = QNetwork()
    target = QNetwork()

    sync_target(online, target)

    online_state = online.state_dict()
    target_state = target.state_dict()

    assert online_state.keys() == target_state.keys()

    for key in online_state:
        assert torch.equal(online_state[key], target_state[key]), key


def test_update_with_unfinished_games_runs():
    torch.manual_seed(0)

    online = QNetwork()
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

    observations, actions, rewards, next_observations, _, next_legal = make_batch()
    dones = np.zeros(16, dtype=np.float32)

    loss = dqn_update(
        online,
        target,
        optimizer,
        (observations, actions, rewards, next_observations, dones, next_legal),
    )

    assert np.isfinite(loss)


def test_network_learns_to_predict_the_reward():
    torch.manual_seed(0)

    online = QNetwork()
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    batch = make_batch()

    for _ in range(500):
        dqn_update(online, target, optimizer, batch)

    observations = torch.as_tensor(batch[0])
    actions = torch.as_tensor(batch[1])

    with torch.no_grad():
        predicted = online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

    # every game ends on the shot, so the target is just the reward, -1.
    assert torch.allclose(predicted, torch.full((16,), -1.0), atol=0.2)