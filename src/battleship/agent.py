import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, observations):
        return self.layers(observations).flatten(start_dim=1)


def choose_action(network, observation, legal, epsilon, rng):
    legal_cells = np.flatnonzero(legal)

    if len(legal_cells) == 0:
        raise ValueError("no actions remaining")

    if rng.random() < epsilon:
        return int(rng.choice(legal_cells))

    device = next(network.parameters()).device

    observation = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    was_training = network.training
    network.eval()

    try:
        with torch.no_grad():
            q_values = network(observation)[0]

            legal_mask = torch.as_tensor(
                legal,
                dtype=torch.bool,
                device=device
            )

            q_values[~legal_mask] = -float("inf")

            return int(q_values.argmax().item())
    finally:
        network.train(was_training)


def compute_targets(
    online,
    target,
    rewards,
    next_observations,
    dones,
    next_legal,
    gamma,
):
    with torch.no_grad():
        targets = rewards.clone()
        nonterminal = ~dones.bool()

        if nonterminal.any():
            next_obs = next_observations[nonterminal]
            legal = next_legal[nonterminal].bool()

            if not legal.any(dim=1).all():
                raise ValueError("no legal action left")

            online_q = online(next_obs)
            online_q = online_q.masked_fill(
                ~legal, -float("inf")
            )
            best_actions = online_q.argmax(
                dim=1, keepdim=True
            )

            target_q = target(next_obs).gather(
                1, best_actions
            ).squeeze(1)

            targets[nonterminal] += gamma * target_q

        return targets


def dqn_update(online, target, optimizer, batch, gamma=0.99, max_grad_norm=10.0):
    device = next(online.parameters()).device

    observations, actions, rewards, next_observations, dones, next_legal = (
        torch.as_tensor(array, device=device) for array in batch
    )

    q_values = online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    targets = compute_targets(
        online, target, rewards, next_observations, dones, next_legal, gamma
    )

    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()


def sync_target(online, target):
    target.load_state_dict(online.state_dict())