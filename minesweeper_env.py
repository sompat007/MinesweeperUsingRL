"""
Custom Minesweeper environment, written from scratch (no gym dependency required,
but the API mirrors the standard reset()/step() gym pattern so it's easy to
swap in gymnasium later if you want).

Key design decisions (read this before touching the reward / obs logic):

1. First-click safety: the very first cell you reveal can never be a mine,
   and its neighbors are also kept mine-free. Mines are placed *after* the
   first action, excluding that cell and its 8 neighbors. This matches how
   real Minesweeper implementations behave and avoids "unfair" instant losses
   that would otherwise dominate early training with pure noise.

2. Observation: an (11, rows, cols) tensor -
       channels 0-8: one-hot "this revealed cell shows number k"
       channel 9:    "this cell is still unrevealed"
       channel 10:   constant plane = fraction of board revealed so far
   This is the standard representation used in most Minesweeper-RL papers
   (e.g. channels-as-one-hot-numbers) because it lets a CNN reason locally
   about numbers the way a human does, without needing to learn an embedding
   for the "unrevealed" token from scratch.

3. Action space: n_rows * n_cols discrete actions, one per cell = "reveal
   this cell". No flagging actions - flagging doesn't change what the agent
   can deduce, so it's dropped to keep the action space small. Revealing an
   already-revealed cell is treated as invalid: it's masked out during
   training (see valid_action_mask) but step() also handles it defensively
   with a penalty in case something calls it anyway (e.g. during manual play).

4. Reward shaping: rewards are intentionally dense (small positive reward per
   newly-revealed safe cell) rather than sparse win/lose only. Minesweeper's
   true objective is binary (win/lose) but with a 16x16/40-mine board the
   win signal is too sparse for PPO to bootstrap from scratch in a reasonable
   number of samples. Dense shaping turns "clear more of the board safely"
   into a learnable gradient while keeping the terminal win/lose rewards
   dominant enough that the agent still optimizes for the real objective.
"""

import numpy as np


class MinesweeperEnv:
    def __init__(self, rows=16, cols=16, n_mines=40, seed=None,
                 reward_reveal=0.1, reward_win=5.0, reward_lose=-1.0,
                 reward_invalid=-0.3):
        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.n_cells = rows * cols
        self.n_safe_cells = self.n_cells - self.n_mines

        self.reward_reveal = reward_reveal
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_invalid = reward_invalid

        self.rng = np.random.default_rng(seed)

        # state, set properly in reset()
        self.mines = None
        self.board_numbers = None
        self.revealed = None
        self.first_move = True
        self.done = False
        self.n_revealed = 0

        self.reset()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mines = None
        self.board_numbers = None
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.first_move = True
        self.done = False
        self.n_revealed = 0
        return self._get_obs()

    def step(self, action):
        """action: int in [0, rows*cols). Returns (obs, reward, done, info)."""
        if self.done:
            raise RuntimeError("step() called on an already-finished episode; call reset()")

        r, c = divmod(int(action), self.cols)

        if self.revealed[r, c]:
            # Should never happen if you use valid_action_mask() during
            # training, but handled safely for manual/interactive play.
            return self._get_obs(), self.reward_invalid, False, {"invalid": True}

        if self.first_move:
            self._place_mines(r, c)
            self.first_move = False

        if self.mines[r, c]:
            self.revealed[r, c] = True
            self.done = True
            return self._get_obs(), self.reward_lose, True, {"win": False, "invalid": False}

        newly_revealed = self._reveal_cascade(r, c)
        reward = self.reward_reveal * newly_revealed

        if self.n_revealed >= self.n_safe_cells:
            self.done = True
            reward += self.reward_win
            return self._get_obs(), reward, True, {"win": True, "invalid": False}

        return self._get_obs(), reward, False, {"win": False, "invalid": False}

    def valid_action_mask(self):
        """Boolean array of length rows*cols; True = legal (unrevealed) cell."""
        return (~self.revealed).flatten()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_neighbors(self, r, c):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _place_mines(self, safe_r, safe_c):
        excluded = set(self._get_neighbors(safe_r, safe_c))
        excluded.add((safe_r, safe_c))

        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        candidates = [cell for cell in all_cells if cell not in excluded]

        if self.n_mines > len(candidates):
            raise ValueError(
                f"n_mines ({self.n_mines}) too large for board size "
                f"({self.rows}x{self.cols}) with a safe first-click zone."
            )

        chosen_idx = self.rng.choice(len(candidates), size=self.n_mines, replace=False)
        self.mines = np.zeros((self.rows, self.cols), dtype=bool)
        for idx in chosen_idx:
            r, c = candidates[idx]
            self.mines[r, c] = True

        self.board_numbers = np.zeros((self.rows, self.cols), dtype=np.int8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mines[r, c]:
                    self.board_numbers[r, c] = -1
                else:
                    cnt = sum(1 for nr, nc in self._get_neighbors(r, c) if self.mines[nr, nc])
                    self.board_numbers[r, c] = cnt

    def _reveal_cascade(self, r, c):
        """Flood-fill reveal starting at (r, c): reveals (r, c), and if it's a
        0, recursively reveals all neighbors (standard Minesweeper cascade)."""
        stack = [(r, c)]
        count = 0
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr, cc]:
                continue
            self.revealed[cr, cc] = True
            count += 1
            self.n_revealed += 1
            if self.board_numbers[cr, cc] == 0:
                for nr, nc in self._get_neighbors(cr, cc):
                    if not self.revealed[nr, nc] and not self.mines[nr, nc]:
                        stack.append((nr, nc))
        return count

    def _get_obs(self):
        obs = np.zeros((11, self.rows, self.cols), dtype=np.float32)
        if self.board_numbers is None:
            # before first move: everything unrevealed
            obs[9, :, :] = 1.0
            return obs
        revealed_numbers = np.where(self.revealed, self.board_numbers, -1)
        for k in range(9):
            obs[k] = (revealed_numbers == k).astype(np.float32)
        obs[9] = (~self.revealed).astype(np.float32)
        obs[10, :, :] = self.n_revealed / self.n_cells
        return obs

    # ------------------------------------------------------------------
    # Human-readable rendering (for play.py / debugging)
    # ------------------------------------------------------------------
    def render(self):
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if not self.revealed[r, c]:
                    row_chars.append(".")
                elif self.mines is not None and self.mines[r, c]:
                    row_chars.append("*")
                else:
                    n = self.board_numbers[r, c]
                    row_chars.append(" " if n == 0 else str(n))
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
