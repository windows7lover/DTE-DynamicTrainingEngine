# datasets/numeric_seq_metadata.py
from __future__ import annotations
from dataclasses import dataclass, field

from dte.data.base_dataset_metadata import BaseDatasetMetadata


@dataclass
class NumericSeqMetadata(BaseDatasetMetadata):
    """
    Metadata for numeric sequence tasks using:
        0         = pad
        1..X      = valid tokens where X = max_value

    User must provide ONLY:
        - seq_len
        - max_value
        - optional: min_len_seq

    Derived:
        - min_value = 1
        - pad_id = 0
        - vocab_size = max_value + 1
    """

    seq_len: int
    max_value: int

    # Optional
    min_len_seq: int = 2

    # Fixed values (never overridden by config)
    pad_id: int = field(init=False, default=0)
    min_value: int = field(init=False, default=1)

    # Derived
    vocab_size: int = field(init=False)

    def __post_init__(self):
        # Prevent user from passing vocab_size or pad_id via YAML
        if "vocab_size" in self.__dict__:
            raise ValueError("vocab_size must not be provided in the config.")
        if "pad_id" in self.__dict__:
            raise ValueError("pad_id must not be provided in the config.")
        if "min_value" in self.__dict__:
            raise ValueError("min_value must not be provided in the config.")

        # Validate max_value
        if self.max_value < 1:
            raise ValueError("max_value must be >= 1.")

        # Compute vocabulary size
        # pad_id = 0, symbols = 1..max_value → total = max_value + 1
        self.vocab_size = self.max_value + 1

        # Validate sequence constraints
        if self.min_len_seq < 1:
            raise ValueError("min_len_seq must be >= 1")
        if self.min_len_seq > self.seq_len:
            raise ValueError("min_len_seq must be <= seq_len")
