import numpy as np
from torch.utils.data import Dataset
import torch

from dte.data.batch import Batch


class FixedNumpyDataset(Dataset):
    def __init__(self, path, dataset_metadata):
        self.dataset_metadata = dataset_metadata
        arr = np.load(path, allow_pickle=True).item()
        self.inputs = torch.from_numpy(arr["inputs"])
        self.labels = torch.from_numpy(arr["labels"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Return a single sample as a Batch:
            inputs:    [seq_len]
            labels:    [seq_len]
            sample_id: scalar (index)
        """
        return Batch(
            inputs=self.inputs[idx],
            labels=self.labels[idx],
            sample_id=torch.tensor(idx, dtype=torch.long),
        )
