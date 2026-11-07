"""Adaptadores de los conjuntos `.npz` a `torch.utils.data`."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.generate import FlyDataset
from ..evaluation.metrics import sequence_windows


class FlySequenceDataset(Dataset):
    """Devuelve (cuadros (T, S, S) float, objetivo (3,), acción) para cada muestra.
    Las ventanas temporales se calculan una vez y nunca cruzan episodios."""

    def __init__(self, data: FlyDataset, seq_len: int):
        self.patches = torch.from_numpy(np.ascontiguousarray(data.patches))       # uint8
        self.goals = torch.from_numpy(np.ascontiguousarray(data.goals)).float()
        self.actions = torch.from_numpy(np.ascontiguousarray(data.actions)).long()
        self.windows = torch.from_numpy(sequence_windows(data.episode, seq_len)).long()

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, i: int):
        return self.patches[self.windows[i]].float(), self.goals[i], self.actions[i]


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int,
                pin_memory: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      worker_init_fn=seed_worker, generator=generator, pin_memory=pin_memory,
                      drop_last=False)
