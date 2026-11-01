import numpy as np


class ReplayBuffer:
    """Fixed-size memory of transitions. When full, the oldest are overwritten."""

    def __init__(self, capacity, board_size):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        if board_size <= 0:
            raise ValueError("board_size must be positive")

        self.capacity = capacity
        self.board_size = board_size

        self.observations = np.zeros(
            (capacity, 4, board_size, board_size),
            dtype=np.float32
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_legal = np.zeros(
            (capacity, board_size * board_size),
            dtype=bool
        )

        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, done, next_legal):
        i = self.position

        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.dones[i] = done
        self.next_legal[i] = next_legal

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size > self.size:
            raise ValueError("not enough transitions stored to sample")

        indices = rng.choice(self.size, size=batch_size, replace=False)

        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.next_legal[indices],
        )