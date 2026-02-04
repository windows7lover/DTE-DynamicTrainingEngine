from __future__ import annotations

import numpy as np
import warnings
import inspect

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch import from_numpy

from dte.data.batch import Batch
from dte.interfaces import GeneratorProtocol


class OnlineDataset(IterableDataset):
    """
    Infinite online dataset that supports BOTH:

    1. Batched generators (fast path):
         __call__(rng, batch_size) -> (inp[B,L], lbl[B,L])
         __call__(rng, batch_size) -> (inp[B,L], lbl[B,L], sid[B])

    2. Per-sample prototype generators (slow fallback):
         __call__(rng) -> (inp[L], lbl[L])
         __call__(rng) -> (inp[L], lbl[L], sid)

    OnlineDataset chooses the fast batched path whenever possible.
    It warns once if fallback mode is required.
    """

    # ---------------------------------------------------------------
    def __init__(
        self,
        generator_fn: GeneratorProtocol,
        internal_batch_size: int,
        base_seed: int,
        dataset_metadata,
    ):
        super().__init__()
        self.generator_fn = generator_fn
        self.internal_batch_size = 4*int(internal_batch_size)
        self.base_seed = int(base_seed)
        self.dataset_metadata = dataset_metadata

        # Detect batching support robustly (ignores Protocol masking)
        self._batched = self._supports_batched(generator_fn)
        self._warned_fallback = False

    # ---------------------------------------------------------------
    @staticmethod
    def _supports_batched(gen) -> bool:
        """
        IMPORTANT:
        We inspect the actual class __call__ method, not the bound method.

        This avoids Protocol masking, which would incorrectly report
        a single-argument signature even for batched generators.
        """
        try:
            sig = inspect.signature(type(gen).__call__)
            # Expect: (self, rng, batch_size) → 3 parameters
            return len(sig.parameters) >= 3
        except Exception:
            return False

    # ---------------------------------------------------------------
    def _generate_batch(self, rng: np.random.Generator):
        """
        Unified batch generator:
        - Fast path: __call__(rng, batch_size)
        - Slow fallback: repeated __call__(rng)

        Returns:
            inputs_np [B, L]
            labels_np [B, L]
            sid_np    [B] or None
        """
        B = self.internal_batch_size
        L = self.dataset_metadata.seq_len

        # -----------------------------
        # Fast batched path
        # -----------------------------
        if self._batched:
            out = self.generator_fn(rng, B)

            if len(out) == 2:
                inp, lbl = out
                return inp, lbl, None
            elif len(out) == 3:
                inp, lbl, sid = out
                return inp, lbl, sid
            else:
                raise RuntimeError(
                    f"Batched generator returned tuple of length {len(out)} "
                    "but expected 2 or 3."
                )

        # -----------------------------
        # Slow fallback path (prototype generators)
        # -----------------------------
        if not self._warned_fallback:
            self._warned_fallback = True
            warnings.warn(
                f"{type(self.generator_fn).__name__} does not implement "
                "__call__(rng, batch_size). Falling back to per-sample generation. "
                "This is MUCH slower. Consider adding a batched implementation.",
                RuntimeWarning,
                stacklevel=2,
            )

        inputs_list = []
        labels_list = []
        sid_list = []
        sid_present = None

        for i in range(B):
            try:
                out = self.generator_fn(rng)
            except TypeError as e:
                raise RuntimeError(
                    f"{type(self.generator_fn).__name__} requires a batched call "
                    f"__call__(rng, batch_size) but fallback mode was triggered. "
                    f"This means _supports_batched() mis-detected the signature.\n"
                    f"Original error: {e}"
                )

            if sid_present is None:
                sid_present = (len(out) == 3)

            if sid_present:
                inp_np, lbl_np, sid_np = out
                sid_list.append(sid_np)
            else:
                inp_np, lbl_np = out

            inputs_list.append(inp_np)
            labels_list.append(lbl_np)

        inp_np = np.stack(inputs_list, axis=0)
        lbl_np = np.stack(labels_list, axis=0)

        if sid_present:
            sid_np = np.stack(sid_list, axis=0)
        else:
            sid_np = None

        # Sanity checks
        if inp_np.shape != (B, L):
            raise RuntimeError(
                f"Generator returned inputs of shape {inp_np.shape}, "
                f"but expected {(B, L)}"
            )
        if lbl_np.shape != (B, L):
            raise RuntimeError(
                f"Generator returned labels of shape {lbl_np.shape}, "
                f"but expected {(B, L)}"
            )

        return inp_np, lbl_np, sid_np

    # ---------------------------------------------------------------
    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        # Distinct RNG per worker (rank included upstream)
        seed = self.base_seed + worker_id * 123456789
        rng = np.random.default_rng(seed)

        B = self.internal_batch_size
        next_id = 0

        while True:
            inp_np, lbl_np, sid_np = self._generate_batch(rng)

            inputs = from_numpy(inp_np).long()
            labels = from_numpy(lbl_np).long()

            if sid_np is None:
                # default: sequential IDs
                sample_id = torch.arange(next_id, next_id + B, dtype=torch.long)
            else:
                sample_id = torch.from_numpy(sid_np).long()

            next_id += B

            yield Batch(
                inputs=inputs,
                labels=labels,
                sample_id=sample_id,
            )
