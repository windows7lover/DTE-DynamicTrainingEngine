from __future__ import annotations

import torch
from dte.data.batch import Batch


class BatchStream:
    """
    Consumes a DataLoader yielding Batch objects and provides
    exactly-n sample streaming by slicing across batches.

    - One stream == one loader (train OR eval)
    - Resetting restarts the iterator and drops cached samples
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

        self._it = None
        self._cache_inputs = None
        self._cache_labels = None
        self._cache_sample_id = None
        self._pos = 0

    def reset(self):
        self._it = iter(self.loader)
        self._cache_inputs = None
        self._cache_labels = None
        self._cache_sample_id = None
        self._pos = 0

    def _refill(self):
        if self._it is None:
            self._it = iter(self.loader)

        try:
            batch = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            batch = next(self._it)

        if not isinstance(batch, Batch):
            raise TypeError(f"Expected Batch, got {type(batch)}")

        self._cache_inputs = batch.inputs
        self._cache_labels = batch.labels
        self._cache_sample_id = batch.sample_id
        self._pos = 0

    def next_n_samples(self, n: int) -> Batch | None:
        if n <= 0:
            return None

        ins, labs, sids = [], [], []
        remaining = n

        while remaining > 0:
            if self._cache_inputs is None or self._pos >= self._cache_inputs.shape[0]:
                self._refill()

            avail = self._cache_inputs.shape[0] - self._pos
            take = min(remaining, avail)

            a, b = self._pos, self._pos + take
            ins.append(self._cache_inputs[a:b])
            labs.append(self._cache_labels[a:b])
            if self._cache_sample_id is not None:
                sids.append(self._cache_sample_id[a:b])

            self._pos = b
            remaining -= take

        if len(ins) == 1:
            return Batch(
                inputs=ins[0],
                labels=labs[0],
                sample_id=sids[0] if sids else None,
            )

        return Batch(
            inputs=torch.cat(ins, 0),
            labels=torch.cat(labs, 0),
            sample_id=torch.cat(sids, 0) if sids else None,
        )
