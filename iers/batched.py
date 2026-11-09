"""Stacked tensors shared by all seeds of a group: layers and the replay memory.

A separate run is a group with S = 1, so separate and batched training use the same code.
"""

from __future__ import annotations

import numpy as np
import torch

# The source of each stored transition. Kept beside the buffer, never sampled from,
# so it cannot change what a learner sees. -1 marks a row of unknown source, for example one restored from a
# checkpoint written before the tags existed.
SOURCES = ("explore", "policy", "replay", "copy")
SOURCE_CODE = {name: code for code, name in enumerate(SOURCES)}


def stack_linear(weights: list[torch.Tensor], biases: list[torch.Tensor]):
    """Stack nn.Linear parameters into W (S, in, out) and b (S, 1, out) for torch.bmm."""
    W = torch.stack([w.T for w in weights])
    b = torch.stack([v.unsqueeze(0) for v in biases])
    return W, b


def unstack_linear(W: torch.Tensor, b: torch.Tensor, index: int):
    """Inverse of stack_linear for one seed, in nn.Linear shapes."""
    return W[index].T, b[index].squeeze(0)


class StackedReplay:
    """One ring buffer per seed, held in a single (S, capacity, D) tensor.

    Each row is obs | action | reward | next_obs | done. Sampling is uniform and without
    replacement, as in the upstream `random.sample`. Each seed draws from its own generator.
    """

    def __init__(self, S: int, capacity: int, obs_dim: int, act_dim: int, device, dtype,
                 touch: bool = False):
        self.S, self.capacity = S, capacity
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.D = 2 * obs_dim + act_dim + 2
        self.device, self.dtype = torch.device(device), dtype
        self.data = torch.empty((S, capacity, self.D), device=self.device, dtype=dtype)
        if touch:  # Commit the memory now, so that "does it fit" is tested at the start.
            self.data.fill_(0)
        self.size = np.zeros(S, dtype=np.int64)
        self.total = np.zeros(S, dtype=np.int64)  # rows ever inserted
        self._seed_index = torch.arange(S, device=self.device).unsqueeze(1)
        self._step_index = torch.arange(S, device=self.device)
        self._seeds_np = np.arange(S)
        self.tags = np.full((S, capacity), -1, dtype=np.int8)
        self.composition = np.zeros((S, len(SOURCES)), dtype=np.int64)  # rows of each source in the buffer now
        self.evicted = np.zeros((S, len(SOURCES)), dtype=np.int64)  # rows of each source overwritten so far
        self.last_idx: np.ndarray | None = None

    def _tag(self, seeds: np.ndarray, positions: np.ndarray, codes: np.ndarray) -> None:
        """Record the source of rows written at these positions, and count the rows they overwrite."""
        old = self.tags[seeds, positions]
        gone = old >= 0
        np.add.at(self.evicted, (seeds[gone], old[gone]), 1)
        np.add.at(self.composition, (seeds[gone], old[gone]), -1)
        self.tags[seeds, positions] = codes
        new = codes >= 0
        np.add.at(self.composition, (seeds[new], codes[new]), 1)

    def add_step(self, rows: np.ndarray, codes: np.ndarray | None = None) -> None:
        """Insert one row per seed. Failed seeds still get a row, which they never use."""
        ptr_np = self.total % self.capacity
        ptr = torch.as_tensor(ptr_np, device=self.device)
        self.data[self._step_index, ptr] = torch.as_tensor(rows, device=self.device, dtype=self.dtype)
        codes = np.full(self.S, -1, dtype=np.int8) if codes is None else np.asarray(codes, dtype=np.int8)
        self._tag(self._seeds_np, ptr_np, codes)
        self.total += 1
        self.size = np.minimum(self.total, self.capacity)

    def add_many(self, seed: int, rows: np.ndarray, code: int = -1) -> None:
        """Insert many rows for one seed, for DUP."""
        n = len(rows)
        positions_np = (self.total[seed] + np.arange(n)) % self.capacity
        positions = torch.as_tensor(positions_np, device=self.device)
        self.data[seed, positions] = torch.as_tensor(rows, device=self.device, dtype=self.dtype)
        self._tag(np.full(n, seed), positions_np, np.full(n, code, dtype=np.int8))
        self.total[seed] += n
        self.size[seed] = min(self.total[seed], self.capacity)

    def sample(self, rngs: list[np.random.Generator], batch: int) -> torch.Tensor:
        """Return a (S, batch, D) tensor. Seeds that are not training yet get valid rows too."""
        idx = np.empty((self.S, batch), dtype=np.int64)
        for i, rng in enumerate(rngs):
            idx[i] = rng.choice(int(self.size[i]), batch, replace=False, shuffle=False)
        self.last_idx = idx
        return self.data[self._seed_index, torch.as_tensor(idx, device=self.device)]

    def sampled_source_counts(self) -> np.ndarray:
        """(S, 4) counts of each source in the last sampled minibatch, in the order of SOURCES."""
        codes = np.take_along_axis(self.tags, self.last_idx, axis=1)
        return np.stack([(codes == c).sum(axis=1) for c in range(len(SOURCES))], axis=1)

    def tags_since(self, seed: int, start_total: int) -> np.ndarray:
        """Source tags of the rows rows_since returns, in the same order."""
        positions = np.arange(start_total, int(self.total[seed])) % self.capacity
        return self.tags[seed, positions].copy()

    def rows_since(self, seed: int, start_total: int) -> np.ndarray:
        """Rows with insertion number in [start_total, total), oldest first, for checkpoints."""
        end_total = int(self.total[seed])
        if end_total - start_total > self.capacity:
            raise RuntimeError("More rows than the buffer capacity were added between checkpoints.")
        positions = (np.arange(start_total, end_total)) % self.capacity
        return self.data[seed, torch.as_tensor(positions, device=self.device)].cpu().numpy()

    def load_rows(self, seed: int, rows: np.ndarray, total: int, tags: np.ndarray | None = None) -> None:
        """Rebuild one seed's ring from its most recent rows, for resuming. tags=None marks them unknown."""
        keep = rows[-self.capacity:]
        first = total - len(keep)
        positions = (first + np.arange(len(keep))) % self.capacity
        self.data[seed, torch.as_tensor(positions, device=self.device)] = torch.as_tensor(
            keep, device=self.device, dtype=self.dtype
        )
        self.tags[seed] = -1
        if tags is not None:
            self.tags[seed, positions] = np.asarray(tags, dtype=np.int8)[-len(keep):] if len(keep) else []
        known = self.tags[seed][self.tags[seed] >= 0]
        self.composition[seed] = np.bincount(known, minlength=len(SOURCES))[:len(SOURCES)]
        self.total[seed] = total
        self.size[seed] = min(total, self.capacity)
