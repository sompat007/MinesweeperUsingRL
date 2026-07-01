"""
Evaluate (and optionally watch, move by move) a trained Minesweeper PPO agent.

Usage:
    python play.py --checkpoint checkpoints/latest.pt --episodes 200
    python play.py --checkpoint checkpoints/latest.pt --episodes 1 --render --sleep 0.3
"""

import argparse
import time

import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from networks import ActorCritic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--render", action="store_true", help="print the board after every move")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between moves when rendering")
    p.add_argument("--deterministic", action="store_true",
                    help="pick the highest-probability legal action instead of sampling")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    rows, cols, mines = train_args["rows"], train_args["cols"], train_args["mines"]

    network = ActorCritic(
        rows=rows, cols=cols, in_channels=11,
        channels=train_args.get("channels", 64), n_blocks=train_args.get("n_blocks", 4),
    ).to(device)
    network.load_state_dict(ckpt["model"])
    network.eval()

    print(f"Loaded checkpoint trained for {ckpt.get('global_step', '?')} steps "
          f"on a {rows}x{cols} board with {mines} mines.")

    env = MinesweeperEnv(rows=rows, cols=cols, n_mines=mines)

    wins = 0
    returns, lengths, cells_revealed_frac = [], [], []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        if args.render:
            print(f"\n=== Episode {ep + 1} ===")

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(env.valid_action_mask(), dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = network.forward(obs_t)
                masked_logits = logits.masked_fill(~mask_t, -1e8)
                if args.deterministic:
                    action = masked_logits.argmax(dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = dist.sample()

            obs, reward, done, info = env.step(action.item())
            ep_return += reward
            steps += 1

            if args.render:
                print(env.render())
                print(f"reward={reward:.2f} total={ep_return:.2f}")
                if args.sleep > 0:
                    time.sleep(args.sleep)

        win = bool(info.get("win", False))
        wins += int(win)
        returns.append(ep_return)
        lengths.append(steps)
        cells_revealed_frac.append(env.n_revealed / env.n_safe_cells)

        if args.render:
            print("WIN" if win else "LOSS", f"| steps={steps} | return={ep_return:.2f}")

    win_rate = wins / args.episodes
    print("\n=== Summary over {} episodes ===".format(args.episodes))
    print(f"Win rate:              {win_rate:.1%}")
    print(f"Mean episode return:   {np.mean(returns):.2f}")
    print(f"Mean episode length:   {np.mean(lengths):.1f}")
    print(f"Mean board cleared %:  {np.mean(cells_revealed_frac):.1%}  (on losses this is partial progress)")


if __name__ == "__main__":
    main()
