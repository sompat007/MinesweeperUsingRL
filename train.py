"""
Train a PPO agent to play Minesweeper.

Usage:
    python train.py
    python train.py --rows 9 --cols 9 --mines 10 --total-timesteps 2000000
    python train.py --resume checkpoints/latest.pt

A note on difficulty and training time (read this before you run it):
16x16/40-mine "Intermediate" Minesweeper is genuinely hard for RL from
scratch - even strong human players lose a meaningful fraction of games, and
optimal play requires probability reasoning PPO has to discover purely from
trial and error. Expect on the order of tens of millions of environment
steps before you see a non-trivial win rate, and don't be surprised if early
training (first ~1-2M steps) looks like it's barely improving - that's
normal while the CNN is still learning basic "don't reveal cells next to
numbers that already have all their mines flagged"-type deductions.

If you want faster feedback while iterating on the code, run with
`--rows 9 --cols 9 --mines 10` (Beginner) first - it converges to a
reasonable win rate much faster and is a good way to sanity-check that
training is working before committing to a long 16x16 run.
"""

import argparse
import os
import time

import numpy as np
import torch

from vec_env import VecMinesweeper
from networks import ActorCritic
from ppo import PPO, RolloutBuffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--mines", type=int, default=40)

    p.add_argument("--total-timesteps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env, per update")

    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--target-kl", type=float, default=0.03)

    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=4)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=20)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = VecMinesweeper(
        num_envs=args.num_envs, rows=args.rows, cols=args.cols,
        n_mines=args.mines, seed=args.seed,
    )
    n_actions = args.rows * args.cols
    obs_shape = (11, args.rows, args.cols)

    network = ActorCritic(
        rows=args.rows, cols=args.cols, in_channels=11,
        channels=args.channels, n_blocks=args.n_blocks,
    ).to(device)

    ppo = PPO(
        network, device,
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches, target_kl=args.target_kl,
    )

    start_update = 1
    global_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        network.load_state_dict(ckpt["model"])
        ppo.optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("update", 1)
        print(f"Resumed from {args.resume} at global_step={global_step}")

    buffer = RolloutBuffer(args.num_steps, args.num_envs, obs_shape, n_actions, device)

    obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=device)
    done = torch.zeros(args.num_envs, device=device)

    # Rolling stats over recently finished episodes, for logging.
    recent_returns, recent_wins, recent_lengths = [], [], []

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            mask = torch.as_tensor(envs.get_masks(), dtype=torch.bool, device=device)

            with torch.no_grad():
                action, log_prob, _, value = network.get_action_and_value(obs, mask)

            next_obs_np, reward, next_done_np, infos = envs.step(action.cpu().numpy())

            buffer.add(step, obs, action, mask, log_prob, value, reward, done)

            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            for info in infos:
                if "episode_return" in info:
                    recent_returns.append(info["episode_return"])
                    recent_lengths.append(info["episode_length"])
                    recent_wins.append(1.0 if info.get("win") else 0.0)

        with torch.no_grad():
            mask = torch.as_tensor(envs.get_masks(), dtype=torch.bool, device=device)
            _, _, _, last_value = network.get_action_and_value(obs, mask)

        stats = ppo.update(buffer, last_value, done)

        if update % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            win_rate = np.mean(recent_wins[-500:]) if recent_wins else float("nan")
            mean_return = np.mean(recent_returns[-500:]) if recent_returns else float("nan")
            mean_length = np.mean(recent_lengths[-500:]) if recent_lengths else float("nan")
            print(
                f"update {update}/{num_updates} | step {global_step:,} | "
                f"sps {sps} | win_rate(last500) {win_rate:.3f} | "
                f"mean_return {mean_return:.2f} | mean_len {mean_length:.1f} | "
                f"pg_loss {stats['pg_loss']:.4f} | v_loss {stats['v_loss']:.4f} | "
                f"entropy {stats['entropy']:.3f} | KL {stats['approx_kl']:.4f} | "
                f"explained_var {stats['explained_variance']:.3f}"
            )

        if update % args.save_interval == 0 or update == num_updates:
            ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
            torch.save({
                "model": network.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "global_step": global_step,
                "update": update + 1,
                "args": vars(args),
            }, ckpt_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
