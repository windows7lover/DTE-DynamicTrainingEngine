from __future__ import annotations

import numpy as np
from dte.interfaces import GeneratorProtocol


class SortGenerator(GeneratorProtocol):
    """
    Fully vectorized sorting dataset generator.

    dataset_metadata must define:
        - seq_len      : int
        - pad_id       : int
        - min_len_seq  : int
        - min_value    : int
        - max_value    : int

    Behaviour (per sample):
        - Sample n ~ Uniform{min_len_seq, ..., seq_len}
        - Sample seq of length n with values in [min_value, max_value]
        - Inputs: seq padded with pad_id to length seq_len
        - Labels: sorted(seq) padded with pad_id to length seq_len

    This version generates a whole batch at once:
        __call__(rng, batch_size) -> (inputs[B, L], labels[B, L])
    """

    def __init__(self, dataset_metadata, seed: int | None = None):
        self.dataset_metadata = dataset_metadata
        self.seed = seed

        # Check required fields exist in dataset_metadata
        for field in ["seq_len", "pad_id", "min_len_seq", "min_value", "max_value"]:
            assert hasattr(dataset_metadata, field), f"dataset_metadata missing required field: {field}"

    def __call__(self, rng: np.random.Generator, batch_size: int):
        """
        Generate `batch_size` samples at once.

        Args:
            rng         : np.random.Generator
            batch_size  : int

        Returns:
            inputs_np : np.ndarray[int64] of shape [B, L]
            labels_np : np.ndarray[int64] of shape [B, L]
        """
        assert isinstance(rng, np.random.Generator), "SortGenerator expects a NumPy Generator."

        B = int(batch_size)
        L = int(self.dataset_metadata.seq_len)

        pad_id = int(self.dataset_metadata.pad_id)
        min_len = int(self.dataset_metadata.min_len_seq)
        min_val = int(self.dataset_metadata.min_value)
        max_val = int(self.dataset_metadata.max_value)

        # --------------------------------------------------------------
        # 1) Sample sequence lengths for each sample (vectorized)
        # --------------------------------------------------------------
        lengths = rng.integers(
            low=min_len,
            high=L + 1,          # exclusive upper bound
            size=B,
            dtype=np.int64,
        )  # shape [B]

        # --------------------------------------------------------------
        # 2) Sample raw full-length sequences [B, L]
        # --------------------------------------------------------------
        raw = rng.integers(
            low=min_val,
            high=max_val + 1,    # exclusive upper bound
            size=(B, L),
            dtype=np.int64,
        )  # shape [B, L]

        # --------------------------------------------------------------
        # 3) Create mask for valid positions per sample
        #    mask[i, j] = True  if j < lengths[i]
        #                = False otherwise
        # --------------------------------------------------------------
        arange_L = np.arange(L, dtype=np.int64)      # [L]
        mask = arange_L[None, :] < lengths[:, None]  # [B, L] bool

        # --------------------------------------------------------------
        # 4) Build inputs: pad_id everywhere, then fill valid positions
        # --------------------------------------------------------------
        inputs = np.full((B, L), pad_id, dtype=np.int64)
        inputs[mask] = raw[mask]

        # --------------------------------------------------------------
        # 5) Build labels:
        #    - For each row, we only want to sort the first `n` valid positions.
        #    - Strategy:
        #        a) For invalid positions, set a sentinel > max_val
        #        b) Sort row-wise -> valid values come first, then sentinels
        #        c) Take first `lengths[i]` sorted entries and pad rest with pad_id
        # --------------------------------------------------------------
        sentinel = np.int64(max_val + 1)

        tmp = raw.copy()
        tmp[~mask] = sentinel           # invalid positions become large

        sorted_full = np.sort(tmp, axis=1)  # [B, L], ascending

        labels = np.full((B, L), pad_id, dtype=np.int64)

        # Valid positions in sorted_full are at columns j < lengths[i]
        valid_sorted_mask = arange_L[None, :] < lengths[:, None]  # [B, L] bool

        labels[valid_sorted_mask] = sorted_full[valid_sorted_mask]

        return inputs, labels
