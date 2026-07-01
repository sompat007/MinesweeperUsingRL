# Minesweeper PPO

A from-scratch PPO implementation (PyTorch, no RL library dependencies) that
learns to play Minesweeper through self-play against the environment.

## Project layout

```
minesweeper_env.py   Custom Minesweeper environment (reset/step/render API)
networks.py           CNN actor-critic (residual blocks + policy/value heads)
vec_env.py            Simple synchronous vectorized env wrapper for parallel rollouts
ppo.py                PPO core: rollout buffer, GAE, clipped surrogate update
train.py               Training loop / entry point
play.py                 Load a checkpoint, evaluate win rate, optionally watch it play
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Default: 16x16 board, 40 mines (Intermediate)
python train.py

# Faster iteration loop for testing changes: Beginner difficulty
python train.py --rows 9 --cols 9 --mines 10 --total-timesteps 3000000

# Resume from a checkpoint
python train.py --resume checkpoints/latest.pt
```

Progress is printed every update: step count, steps/sec, rolling win rate,
mean return, and PPO diagnostics (policy loss, value loss, entropy, KL,
explained variance). Checkpoints save to `checkpoints/latest.pt` periodically.

## Evaluation

```bash
# Win rate over 200 games, no rendering
python play.py --checkpoint checkpoints/latest.pt --episodes 200

# Watch a single game move-by-move in the terminal
python play.py --checkpoint checkpoints/latest.pt --episodes 1 --render --sleep 0.3
```

## How it works

**Environment** (`minesweeper_env.py`): standard Minesweeper rules, with the
usual "first click is always safe" guarantee — mines are placed only after
the first reveal, excluding that cell and its neighbors. The action space
is one discrete action per cell ("reveal this cell"); there's no flagging
action, since flagging doesn't add any information the agent can't already
get from the numbers. The observation is an 11-channel `rows x cols` tensor:
9 one-hot channels for revealed numbers 0–8, 1 channel for "still
unrevealed", and 1 channel tracking overall board progress.

**Network** (`networks.py`): a small residual CNN. The policy head outputs
one logit per cell; the value head pools down to a scalar. Illegal actions
(already-revealed cells) are masked to ~0 probability before sampling, which
keeps the agent from wasting samples on actions that can never be correct.

**PPO** (`ppo.py`): textbook PPO — GAE for advantages, the clipped surrogate
objective, a clipped value loss, an entropy bonus for exploration, minibatch
epochs per rollout, gradient clipping, and an optional KL-based early stop
per update.

## Difficulty and training time

16x16 with 40 mines ("Intermediate") is a genuinely hard RL problem — even
strong human players lose a meaningful fraction of games, and good play
requires probability reasoning the agent has to discover entirely through
trial and error, with no supervision. It takes on the order of tens of
millions of environment steps to reach a non-trivial win rate, and early
training tends to look flat for a while before that.

For quicker feedback while iterating on the code itself, Beginner difficulty
(`--rows 9 --cols 9 --mines 10`) trains much faster and is a good way to
confirm the pipeline is learning before committing to a long Intermediate
run. A **curriculum** approach also works: train on Beginner first, then
`--resume` that checkpoint while switching to `--rows 16 --cols 16 --mines
40`. Note that the network's conv layers transfer across board sizes fine
since it's fully convolutional up to the value head's flatten — but the
value head's `Linear` layer is sized for a specific `rows*cols`, so a
straight `--resume` across board sizes will fail to load that layer. The
clean fix is reinitializing just the value head's final `Linear` after
loading the rest of the state dict.

## Possible extensions

- **Reward shaping tweaks**: a smaller penalty for revealing a cell adjacent
  to a known, fully-flagged mine cluster, or a shaping bonus for revealing
  cells with lower true mine probability (computable from the board state),
  to speed up early learning.
- **Larger/deeper network**: not strictly needed here (state is fully
  observable and Markovian, so no frame stacking is required), but a bigger
  `channels`/`n_blocks` could help on Expert-level boards.
- **Curriculum learning** across board sizes/mine densities (see above).
- **Constraint-satisfaction assist**: many strong Minesweeper agents combine
  RL/heuristics with an exact CSP solver for "100% certain" moves, falling
  back to the learned policy only for genuinely probabilistic guesses.
