
import numpy as np
import pytest
import torch

from battleship.agent import QNetwork, choose_action


def test_network_gives_one_value_per_cell():
    network = QNetwork()
    batch = torch.zeros(5, 4, 6, 6)

    assert network(batch).shape == (5, 36)


def test_same_network_works_on_a_bigger_board():
    network = QNetwork()
    batch = torch.zeros(2, 4, 10, 10)

    assert network(batch).shape == (2, 100)


def test_greedy_choice_is_always_legal():
    torch.manual_seed(0)
    network = QNetwork()
    observation = np.zeros((4, 6, 6), dtype=np.float32)
    rng = np.random.default_rng(0)

    legal = np.zeros(36, dtype=bool)
    legal[[3, 17, 30]] = True

    for _ in range(50):
        assert choose_action(network, observation, legal, 0.0, rng) in (3, 17, 30)


def test_greedy_choice_selects_highest_valued_legal_action():
    class FixedNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, observations):
            return torch.tensor(
                [[100.0, 5.0, 20.0, 30.0]],
                device=observations.device
            )

    network = FixedNetwork()
    observation = np.zeros((4, 2, 2), dtype=np.float32)
    legal = np.array([False, True, True, False])

    action = choose_action(
        network,
        observation,
        legal,
        0.0,
        np.random.default_rng(0)
    )

    assert action == 2


def test_random_choice_is_always_legal():
    network = QNetwork()
    observation = np.zeros((4, 6, 6), dtype=np.float32)
    rng = np.random.default_rng(1)

    legal = np.zeros(36, dtype=bool)
    legal[[8, 9]] = True

    for _ in range(50):
        assert choose_action(network, observation, legal, 1.0, rng) in (8, 9)


def test_single_legal_cell_is_chosen():
    network = QNetwork()
    observation = np.zeros((4, 6, 6), dtype=np.float32)
    legal = np.zeros(36, dtype=bool)
    legal[22] = True

    assert choose_action(network, observation, legal, 0.0, np.random.default_rng(0)) == 22


def test_choose_action_raises_when_no_legal_actions():
    network = QNetwork()
    observation = np.zeros((4, 6, 6), dtype=np.float32)
    legal = np.zeros(36, dtype=bool)

    with pytest.raises(ValueError, match="no actions remaining"):
        choose_action(network, observation, legal, 0.0, np.random.default_rng(0))


def test_choose_action_preserves_network_mode():
    network = QNetwork()
    observation = np.zeros((4, 6, 6), dtype=np.float32)
    legal = np.ones(36, dtype=bool)
    rng = np.random.default_rng(0)

    network.train()
    choose_action(network, observation, legal, 0.0, rng)
    assert network.training

    network.eval()
    choose_action(network, observation, legal, 0.0, rng)
    assert not network.training
