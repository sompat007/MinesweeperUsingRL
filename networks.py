"""
Actor-Critic network for Minesweeper.

Minesweeper is a spatial, local-reasoning task (a cell's number only tells
you about its 8 neighbors), so a CNN with a few residual blocks is a natural
fit - it's the same architectural family used for Go/chess policy networks,
just much smaller since we don't need anywhere near that capacity here.

The network is fully convolutional up to the policy head, which outputs one
logit per board cell (i.e. "how good is it to reveal this cell"). Because
the board is a fixed rows x cols grid for a given training run, the value
head flattens down to a small MLP.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ActorCritic(nn.Module):
    def __init__(self, rows=16, cols=16, in_channels=11, channels=64, n_blocks=4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])

        # Policy head: 1x1 conv down to a single channel -> one logit per cell.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Value head: pool down to a small vector -> scalar.
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * rows * cols, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """x: (B, in_channels, rows, cols) -> (logits (B, rows*cols), value (B,))"""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        logits = self.policy_head(h).flatten(1)          # (B, rows*cols)
        value = self.value_head(h).squeeze(-1)            # (B,)
        return logits, value

    def get_action_and_value(self, x, action_mask, action=None):
        """
        x: (B, C, H, W) observation batch
        action_mask: (B, rows*cols) bool tensor, True = legal action
        action: optional (B,) tensor of actions to evaluate (for PPO update).
                If None, an action is sampled from the (masked) policy.

        Returns: action, log_prob, entropy, value
        """
        logits, value = self.forward(x)
        # Push illegal actions to ~-inf so they get ~0 probability, without
        # producing NaNs the way -inf can under some softmax implementations.
        masked_logits = logits.masked_fill(~action_mask, -1e8)
        dist = torch.distributions.Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value
