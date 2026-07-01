# Minesweeper PPO

A from-scratch PPO implementation (PyTorch, no RL library dependencies) that
learns to play Minesweeper via self-play against the environment.

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

## Train

```bash
# Default: 16x16 board, 40 mines (Intermediate)
python train.py

# Faster iteration loop while you're testing changes: Beginner difficulty
python train.py --rows 9 --cols 9 --mines 10 --total-timesteps 3000000

# Resume from a checkpoint
python train.py --resume checkpoints/latest.pt
```

Progress is printed every update: step count, steps/sec, rolling win rate,
mean return, PPO diagnostics (policy loss, value loss, entropy, KL,
explained variance). Checkpoints save to `checkpoints/latest.pt` periodically.

## Evaluate / watch

```bash
# Win-rate over 200 games, no rendering
python play.py --checkpoint checkpoints/latest.pt --episodes 200

# Watch a single game move-by-move in the terminal
python play.py --checkpoint checkpoints/latest.pt --episodes 1 --render --sleep 0.3
```

## How it works

**Environment** (`minesweeper_env.py`): standard Minesweeper rules, with the
usual "first click is always safe" guarantee (mines are placed only after
the first reveal, excluding that cell and its neighbors). The action space
is one discrete action per cell ("reveal this cell") — no flagging, since it
doesn't add any information the agent can't already get from the numbers.
Observation is an 11-channel `rows x cols` tensor: 9 one-hot channels for
revealed numbers 0–8, 1 channel for "still unrevealed", and 1 channel
tracking overall board progress.

**Network** (`networks.py`): a small residual CNN. The policy head outputs
one logit per cell; the value head pools down to a scalar. Illegal actions
(already-revealed cells) are masked to ~0 probability before sampling —
this is important, since without masking the agent wastes a large fraction
of its samples on actions that can never be correct.

**PPO** (`ppo.py`): textbook PPO — GAE for advantages, the clipped surrogate
objective, a clipped value loss, an entropy bonus for exploration, minibatch
epochs per rollout, gradient clipping, and an optional KL-based early stop
per update.

## A note on difficulty and training time

16x16 with 40 mines ("Intermediate") is a genuinely hard RL problem — even
strong human players lose a meaningful fraction of games, and good play
requires probability reasoning the agent has to discover entirely from trial
and error, with no supervision. Expect to need on the order of tens of
millions of environment steps before you see a non-trivial win rate, and
expect early training to look flat for a while.

If you want quick feedback while iterating on the code itself, run
`--rows 9 --cols 9 --mines 10` (Beginner) first — it trains much faster and
is a good way to confirm the pipeline is actually learning before committing
to a long Intermediate run. You can also try a **curriculum**: train on
Beginner first, then `--resume` that checkpoint while switching to
`--rows 16 --cols 16 --mines 40` (note: the network's conv layers transfer
across board sizes fine since it's fully convolutional up to the value
head's flatten — but the value head's `Linear` layer is sized for a specific
`rows*cols`, so a straight `--resume` across board sizes will fail to load
that layer. If you want to do this, the cleanest fix is to zero out/reinit
just the value head's final `Linear` after loading the rest of the
state dict).

## Ideas for extending this

- **Reward shaping tweaks**: penalize revealing a cell adjacent to a known,
  fully-flagged mine cluster less harshly, or add a small shaping bonus for
  revealing cells with lower true mine probability (computable from the
  board state) to speed up early learning.
- **Larger/deeper network** or **frame stacking** isn't needed here (state
  is fully observable and Markovian), but a bigger `channels`/`n_blocks`
  might help on Expert-level boards.
- **Curriculum learning** across board sizes/mine densities (see above).
- **Constraint-satisfaction assist**: many strong Minesweeper agents combine
  RL/heuristics with an exact CSP solver for "100% certain" moves, and only
  fall back to the learned policy for genuinely probabilistic guesses.
