# Minesweeper PPO

A Proximal Policy Optimization (PPO) agent that learns to play Minesweeper,
implemented from scratch in PyTorch with no RL library dependencies.

## Project layout

```
minesweeper_env.py    Custom Minesweeper environment (reset/step/render API)
networks.py            CNN actor-critic (residual blocks + policy/value heads)
vec_env.py             Synchronous vectorized env wrapper for parallel rollouts
ppo.py                 PPO core: rollout buffer, GAE, clipped surrogate update
train.py                Training loop / entry point
play.py                  Load a checkpoint, evaluate win rate, watch it play
requirements.txt
```

## Why PPO

Minesweeper is a discrete-action, fully-observable, single-agent sequential
decision problem, which fits the standard RL setup well: the board state is
Markovian (everything relevant to the next move is visible on the board),
and there's a natural reward signal (revealing safe cells, winning, hitting
a mine).

PPO specifically is a reasonable default for this kind of problem for a few
reasons:

- **On-policy and stable.** Minesweeper's reward landscape is deceptive —
  early in training almost every action looks equally bad because most of
  them eventually lead to random mine hits. Off-policy value-based methods
  (DQN and variants) can be unstable under this kind of noisy, sparse-ish
  reward without a lot of tuning. PPO's clipped surrogate objective bounds
  how far a single update can move the policy, which keeps training stable
  even when the advantage estimates are noisy.
- **Works naturally with a large discrete action space.** A 16x16 board has
  256 possible actions. PPO with a categorical policy head handles this
  directly; no need for action-value tables or discretization tricks that
  some other methods need.
- **Straightforward action masking.** Most cells are illegal to reveal at
  any given time (they're already revealed). PPO's policy gradient just
  needs the invalid logits pushed to ~0 probability before sampling — this
  integrates cleanly into the categorical distribution and doesn't require
  restructuring the algorithm.
- **Sample efficiency is "good enough" without being fragile.** Model-based
  methods would likely be more sample-efficient here, but Minesweeper's
  transition dynamics (mine layout, cascade reveals) are simple to simulate
  cheaply, so raw sample count is less of a bottleneck than training
  stability. PPO trades some sample efficiency for a much simpler, more
  robust training recipe.

## Environment design

**First-click safety.** Mines are placed only after the first reveal,
excluding that cell and its 8 neighbors. This mirrors standard Minesweeper
implementations and avoids polluting early training with instant, purely
random losses on move one — those provide no learning signal since they
can't be predicted from any information the agent has.

**Action space.** One discrete action per cell ("reveal this cell"), with
no separate flagging action. Flagging is a bookkeeping convenience for human
players; it doesn't reveal any information the agent can't already infer
directly from the numbers on the board, so including it would only enlarge
the action space without adding learnable signal.

**Observation representation.** An 11-channel `rows x cols` tensor:
- 9 channels, one-hot per revealed number (0–8)
- 1 channel marking still-unrevealed cells
- 1 channel giving a constant "fraction of board revealed" plane

One-hot encoding revealed numbers (rather than, say, a single channel with
raw integer values) is standard practice for CNN inputs, since it avoids
implying a false ordinal relationship between the channels — a cell showing
"3" is not "more" than a cell showing "1" in any way that should be encoded
by magnitude.

**Reward shaping.** Minesweeper's true objective is binary (win or lose),
but that signal is far too sparse to bootstrap PPO from a random policy on
a 16x16/40-mine board — most random rollouts hit a mine within a handful of
moves, and a win takes over 200 correct actions in a row. A small positive
reward per newly revealed safe cell turns "clear more of the board safely"
into a dense, learnable gradient, while the terminal win/lose rewards
(dominant in magnitude) still keep the true objective in charge of what
the policy ultimately optimizes for.

## Network architecture

Minesweeper is spatial and locally structured — a cell's number only
constrains its 8 immediate neighbors — which makes a convolutional network
a natural fit, the same reasoning that motivates CNN policy networks in
Go/chess engines, just at much smaller scale here. The network is a small
residual CNN (a handful of ResBlocks) shared between the policy and value
heads:

- **Policy head**: 1x1 convolutions down to a single-channel map, flattened
  to one logit per board cell.
- **Value head**: 1x1 convolution, flatten, then a small MLP down to a
  scalar state-value estimate.

Illegal actions (already-revealed cells) are masked by setting their logits
to a large negative number before softmax, so they receive ~0 probability
without producing NaNs.

## PPO implementation details

Standard PPO components, implemented directly rather than through a library
so every piece is inspectable and modifiable:

- **GAE (Generalized Advantage Estimation)** for computing advantages with
  a bias/variance tradeoff controlled by `gae_lambda`.
- **Clipped surrogate objective** — the core PPO mechanism that prevents
  any single update from moving the policy too far from the data it was
  collected under.
- **Clipped value loss** — stabilizes the value function the same way the
  policy is stabilized.
- **Entropy bonus** — encourages exploration, particularly important early
  on when most of the board looks equally uncertain.
- **Minibatch epochs per rollout** — reuses each batch of collected
  experience for several gradient steps, improving sample efficiency.
- **Gradient clipping** and an **optional KL-based early stop** per update,
  as extra guardrails against destructively large policy updates.
- **Vectorized environments** (`vec_env.py`) — many Minesweeper boards run
  in parallel to collect a diverse rollout batch per update. A simple
  synchronous Python-loop wrapper is used rather than true multiprocessing,
  since environment stepping is cheap pure-numpy code and doesn't need it.

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

Progress is logged every update: step count, steps/sec, rolling win rate,
mean return, and PPO diagnostics (policy loss, value loss, entropy, KL,
explained variance). Checkpoints save to `checkpoints/latest.pt` periodically.

## Evaluation

```bash
# Win rate over 200 games, no rendering
python play.py --checkpoint checkpoints/latest.pt --episodes 200

# Watch a single game move-by-move in the terminal
python play.py --checkpoint checkpoints/latest.pt --episodes 1 --render --sleep 0.3
```

## Difficulty and training time

16x16 with 40 mines is a genuinely hard RL problem — even strong human
players lose a meaningful fraction of games, and good play requires
probability reasoning the agent has to discover entirely through trial and
error. Reaching a non-trivial win rate takes on the order of tens of
millions of environment steps, with training typically looking flat for a
while before that.

Beginner difficulty (`--rows 9 --cols 9 --mines 10`) converges much faster
and is useful for confirming the pipeline is learning correctly before
committing to a long Intermediate run. A curriculum approach also works:
train on Beginner, then `--resume` that checkpoint while switching to
`--rows 16 --cols 16 --mines 40`. The network's convolutional layers
transfer across board sizes cleanly (fully convolutional up to the value
head's flatten), but the value head's final `Linear` layer is sized for a
specific `rows*cols`, so a direct `--resume` across board sizes will fail
to load that one layer — it needs to be reinitialized separately after
loading the rest of the state dict.

## Possible extensions

- **Reward shaping refinements**: smaller penalties near fully-flagged mine
  clusters, or a shaping bonus scaled by true mine probability (computable
  from the board state) to accelerate early learning.
- **Larger network**: state here is fully observable and Markovian, so no
  frame stacking is needed, but a bigger `channels`/`n_blocks` could help
  on Expert-level boards.
- **Curriculum learning** across board sizes and mine densities.
- **Constraint-satisfaction assist**: many strong Minesweeper agents pair
  RL/heuristics with an exact CSP solver for "100% certain" moves, falling
  back to the learned policy only for genuinely probabilistic guesses.
