
import numpy as np
import torch
from torch import nn


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
