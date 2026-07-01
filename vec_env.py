"""
A minimal synchronous vectorized environment wrapper.

PPO is on-policy and benefits a lot from collecting rollouts across many
parallel environments (it de-correlates the data in each batch and lets you
gather thousands of steps per "rollout" quickly). We don't need real
multiprocessing here - Minesweeper's step() is cheap pure-numpy code - so a
simple Python-loop-based vectorized env keeps things dependency-free and easy
to debug, while still giving PPO the batch diversity it wants.

Each individual env auto-resets when it finishes (standard convention for
vectorized RL envs), so training can just keep calling step() forever without
ever having to special-case episode boundaries in the main loop.
"""

import numpy as np
from minesweeper_env import MinesweeperEnv


class VecMinesweeper:
    def __init__(self, num_envs, rows=16, cols=16, n_mines=40, seed=0, **env_kwargs):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.envs = [
            MinesweeperEnv(rows=rows, cols=cols, n_mines=n_mines, seed=seed + i, **env_kwargs)
            for i in range(num_envs)
        ]
        # per-env running episode stats, useful for logging
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int64)

    def reset(self):
        obs = np.stack([env.reset() for env in self.envs], axis=0)
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return obs

    def get_masks(self):
        return np.stack([env.valid_action_mask() for env in self.envs], axis=0)

    def step(self, actions):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(int(a))
            self.episode_returns[i] += reward
            self.episode_lengths[i] += 1

            if done:
                info = dict(info)
                info["episode_return"] = self.episode_returns[i]
                info["episode_length"] = self.episode_lengths[i]
                obs = env.reset()
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )
