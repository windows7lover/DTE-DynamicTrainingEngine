# datasets/numeric_seq_metadata.py
from __future__ import annotations
from dataclasses import dataclass

from dte.data.base_dataset_metadata import BaseDatasetMetadata

@dataclass
class SudokuMetadata(BaseDatasetMetadata):
    """
    Metadata for Sudoku datasets.
    Tokens:
        pad_id   = 0
        blank_id = 1
        digits   = 2..n+1 (representing 1..n)
    """

    sudoku_size: int      # usually 9
    
    # Infered metadata after init
    seq_len: int = None
    pad_id: int = None
    blank_id: int = None
    vocab_size: int = None

    def __post_init__(self):
        # ------------------------------------------------------------------
        # 1. Compute sequence length
        # ------------------------------------------------------------------
        self.seq_len = self.sudoku_size * self.sudoku_size
        
        # ------------------------------------------------------------------
        # 2. Compute vocab size
        #    Tokens are: 0 (pad), 1 (blank), 2...sudoku_size+2 (sudoku digits)
        # ------------------------------------------------------------------
        self.vocab_size = self.sudoku_size+2

        # ------------------------------------------------------------------
        # 3. Assign token IDs
        #    - pad_id   = 0 (usually 10)
        #    - blank_id = 1
        #    - digits 1..9 → 2..10
        #   This is compatible with the hugging face build_sudoku dataset that has not padding
        # ------------------------------------------------------------------
        self.pad_id = 0
        self.blank_id = 1
