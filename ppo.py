"""
PPO implementation from scratch (PyTorch), following the original PPO paper
(Schulman et al. 2017) and the well-known implementation details from
"The 37 Implementation Details of Proximal Policy Optimization"
(Huang et al.), adapted for a discrete action space with action masking.

Pieces implemented here:
  - RolloutBuffer: stores one rollout (num_steps x num_envs) of transitions.
  - compute_gae: Generalized Advantage Estimation for advantages + returns.
  - PPO.update: the clipped surrogate objective, value loss, entropy bonus,
    minibatch epochs, gradient clipping, and (optionally) an early-stopping
    KL check.

Nothing here is Minesweeper-specific - this file would work for any discrete,
action-masked, vectorized environment.
"""

import numpy as np
import torch
import torch.nn as nn


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, n_actions, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, *obs_shape), device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.masks = torch.zeros((num_steps, num_envs, n_actions), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)

    def add(self, step, obs, action, mask, log_prob, value, reward, done):
        self.obs[step] = obs
        self.actions[step] = action
        self.masks[step] = mask
        self.log_probs[step] = log_prob
        self.values[step] = value
        self.rewards[step] = torch.as_tensor(reward, device=self.device)
        self.dones[step] = torch.as_tensor(done, device=self.device, dtype=torch.float32)


def compute_gae(rewards, values, dones, last_value, last_done, gamma=0.99, gae_lambda=0.95):
    """
    rewards, values, dones: (num_steps, num_envs)
    last_value: (num_envs,)  bootstrap value for the state after the last step
    last_done:  (num_envs,)  whether that last state was terminal

    Returns advantages, returns: both (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


class PPO:
    def __init__(
        self,
        network,
        device,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        num_minibatches=4,
        target_kl=None,
    ):
        self.network = network
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl

        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer, last_value, last_done):
        num_steps, num_envs = buffer.rewards.shape

        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_value, last_done, self.gamma, self.gae_lambda,
        )

        # Flatten (num_steps, num_envs, ...) -> (num_steps*num_envs, ...)
        b_obs = buffer.obs.reshape(-1, *buffer.obs.shape[2:])
        b_actions = buffer.actions.reshape(-1)
        b_masks = buffer.masks.reshape(-1, buffer.masks.shape[-1])
        b_log_probs = buffer.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer.values.reshape(-1)

        batch_size = num_steps * num_envs
        minibatch_size = batch_size // self.num_minibatches
        indices = np.arange(batch_size)

        stats = {}
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            approx_kl_list = []
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_advantages = b_advantages[mb_idx]
                # per-minibatch advantage normalization (standard PPO trick,
                # reduces variance in the policy gradient estimate)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    b_obs[mb_idx], b_masks[mb_idx], action=b_actions[mb_idx]
                )

                log_ratio = new_log_prob - b_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    approx_kl_list.append(approx_kl.item())

                # Clipped surrogate policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss (helps stabilize the value function too)
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_value - b_values[mb_idx], -self.clip_coef, self.clip_coef
                )
                v_loss_unclipped = (new_value - b_returns[mb_idx]) ** 2
                v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

            stats["approx_kl"] = float(np.mean(approx_kl_list))
            if self.target_kl is not None and stats["approx_kl"] > self.target_kl:
                # Early stop this update if the policy has moved too far -
                # keeps updates "proximal", which is the whole point of PPO.
                break

        stats["pg_loss"] = float(pg_loss.item())
        stats["v_loss"] = float(v_loss.item())
        stats["entropy"] = float(entropy_loss.item())
        stats["explained_variance"] = explained_variance(b_values, b_returns)
        return stats


def explained_variance(values, returns):
    var_returns = returns.var()
    if var_returns.item() == 0:
        return float("nan")
    return float(1 - (returns - values).var() / var_returns)
