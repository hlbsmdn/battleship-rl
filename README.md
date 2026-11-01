# battleship-rl

A reinforcement learning agent that learns to play Battleship from scratch.
It starts out knowing nothing, shoots at a hidden board, and is trained to win in as few shots as possible.
The agent is scored against simple hand-written bots on the same fixed set of test boards.

**Status: work in progress.** The game, the bots' evaluation setup and all the learning machinery are written and tested.
The training loop and the first learning curve are the next step, so there are no agent results yet.
See [Status](#status) for exactly what is done.

## The rules

This uses Russian Battleship rules:

- Ships are straight lines and **cannot touch each other, not even diagonally.**
- When a shot sinks a ship, the game **announces it** ("sunk"), and the shooter learns that the ship is gone.
- The board size and the fleet are configurable.
  - 6x6 board with the small fleet `[3, 2, 2, 1, 1]` (9 ship cells), used for development.
  - 10x10 board with the full fleet `[4, 3, 3, 2, 2, 2, 1, 1, 1, 1]` (20 ship cells), planned for later.

## How it works

### 1. The game (`game.py`)

`Game(board_size, fleet, rng)` places the ships at random with the no-touch rule and keeps them hidden.

- `game.shoot(row, col)` returns `"miss"`, `"hit"` or `"sunk"`.
  It raises an error for a shot outside the board or a repeated shot.
- `game.shots` is a grid of what the shooter has learned so far:

  | Value | Meaning |
  | ----- | ------- |
  | 0 | unknown |
  | 1 | miss |
  | 2 | hit (ship not yet sunk) |
  | 3 | part of a sunk ship |

- `game.is_over` becomes true when every ship is sunk.
- `game.board` is the hidden truth.

**Design rule: bots and the agent only ever receive `game.shots`.** They never see `game.board`.
This keeps every comparison fair and makes cheating by accident impossible.

### 2. The environment (`env.py`)

`BattleshipEnv` wraps the game so a learning agent can play it.

- **Observation:** the `shots` grid split into four 0/1 layers (unknown, miss, hit, sunk), shape `(4, N, N)`, `float32`.
  The values 0 to 3 are labels, not amounts, and separate layers stop a network from treating "sunk" as three times "miss".
- **Action:** one integer per cell, `row * N + col`.
- **Legal actions:** `legal_actions()` returns a mask of cells that have not been shot yet, so the agent can never repeat a shot.
- **Reward:** **-1 per shot, nothing else.**
  The total reward of a game is minus the number of shots, so maximizing reward means winning as fast as possible.
  There are no tuned bonus numbers.
  Hit and sunk bonuses are a planned experiment, to be tried only if learning stalls, and compared against this baseline.
- **Seeds:** training boards come from seeds below 1,000,000.
  The fixed test boards start at 1,000,000, so training and testing never overlap.

### 3. The agent (`agent.py`, `replay.py`)

- **`QNetwork`** is a small fully convolutional network: three 3x3 convolutions (32 channels, ReLU), then a 1x1 convolution that outputs one Q-value per cell.
  There is no fully connected layer, so the same network runs on any board size without changes.
- **`choose_action`** is epsilon-greedy.
  With probability epsilon it shoots a random legal cell, so it keeps exploring.
  Otherwise it shoots the legal cell with the highest Q-value (illegal cells are set to minus infinity before the argmax).
- **`ReplayBuffer`** is a fixed-size memory of transitions: observation, action, reward, next observation, done, and which cells were still legal afterwards.
  Training samples random batches from it, because shots from the same game look alike and learning from them in order makes training unstable.
- **Double DQN update (`compute_targets`, `dqn_update`, `sync_target`):**
  - For a game that has not ended, the target is `reward + gamma * Q_target(next state, best next cell)`.
    The online network chooses the best legal next cell and the target network scores that choice.
    This split stops Q-values drifting upwards.
  - For a finished game, the target is just the reward.
  - The loss is Huber loss, with gradient clipping.
  - The target network is a frozen copy that is refreshed from the online network every so often.

### 4. Baselines and evaluation (`bots.py`, `scripts/evaluate.py`)

Every player is scored by the same function on the same 1,000 fixed boards (board seeds 1,000,000 to 1,000,999), so results are reproducible and comparable.
A player is anything with `choose_shot(shots, rng)`, so the trained agent will be scored the same way as the bots.

## Results so far

6x6 board, small fleet, 1,000 fixed test boards. Fewer shots is better.

| Player | Mean shots | Median | Std | Best | Worst |
| ------ | ---------- | ------ | --- | ---- | ----- |
| Random bot | 33.3 | 34 | 2.9 | 18 | 36 |
| Hunt bot | not written yet | | | | |
| DQN agent | not trained yet | | | | |

**Why the random bot averages 33.3:** a random shooter takes cells in a random order, and the game ends when the last of the 9 ship cells is hit.
The expected position of the last of 9 marked cells among 36 is `9 * (36 + 1) / (9 + 1) = 33.3`.
The measured result matches the formula, which is a good check that the game, the bot and the evaluation are all correct.
This is the line the learning agent has to get below.

## Project structure

```
battleship-rl/
├── src/battleship/
│   ├── game.py      # ship placement, Game, shooting rules
│   ├── bots.py      # RandomBot
│   ├── env.py       # RL environment: observation, reward, legal-action mask
│   ├── agent.py     # QNetwork, epsilon-greedy choice, Double DQN update
│   └── replay.py    # replay buffer
├── scripts/
│   └── evaluate.py  # scores a bot on the fixed test boards
├── tests/           # one test file per module
└── pyproject.toml
```

## Setup and usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14 (as set in `pyproject.toml`).
PyTorch is installed from the CPU-only index, so the download is small.

```bash
uv sync
uv run pytest                          # run the tests
uv run python scripts/evaluate.py      # score the random bot on 6x6
```

## Tests

The tests cover:

- **Placement:** correct fleet sizes, ships never touch (checked on hundreds of random boards), ships are straight and continuous, the same seed gives the same board.
- **Shooting:** miss, hit and sunk results, repeated and out-of-board shots, game over, and that unshot ships stay hidden in `shots`.
- **Bots and environment:** no repeated shots, plain integer coordinates, observation shape and layers, the legal-action mask, rewards, and that actions after the game is over are rejected.
- **Agent:** the network output shape (also on 10x10), that the greedy choice is the highest-valued *legal* cell, that random exploration visits every legal cell, and that the network's training mode is restored.
- **Replay buffer:** stored transitions stay together after sampling, the oldest are overwritten when it is full, and samples have no repeats.
- **Double DQN:** the online network picks the action and the target network scores it, illegal cells are excluded, finished games do not use future value, and only the online weights change in an update.

## Roadmap

- [x] Game with no-touch placement and sunk calls
- [x] Random bot baseline and fixed-board evaluation
- [x] RL environment with a flat -1 per shot reward
- [x] Q-network, masked action choice, replay buffer, Double DQN update
- [ ] Training loop, saving scores to a CSV, first learning curve on 6x6
- [ ] Hunt bot (checkerboard search, then attack neighbours of a hit) as a harder baseline
- [ ] Charts: learning curve against the bots, distribution of shots per game
- [ ] Reward experiment: add hit and sunk bonuses and compare with the flat reward
- [ ] Scale up to 8x8 and 10x10, and test whether a network trained on 6x6 plays better than random on 10x10

## How this was built

I built this with Claude as a tutor and code reviewer.
Claude wrote first drafts of some parts and reviewed my versions; I rewrote and tested the code and made the design decisions.
The experiments and analysis will be mine.