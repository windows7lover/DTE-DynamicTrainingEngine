# datasets/batch.py

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Batch:
    """
    Fields:
        inputs:     Tensor[B, ...]
        labels:     Tensor[B, ...]
        sample_id:  Tensor[B] or None
    """

    inputs: torch.Tensor
    labels: torch.Tensor
    sample_id: torch.Tensor | None = None

    # -------------------------------------------------------------
    def to(self, device) -> "Batch":
        """Device transfer for all tensor fields."""
        return Batch(
            inputs=self.inputs.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            sample_id=(
                self.sample_id.to(device, non_blocking=True)
                if self.sample_id is not None else None
            )
        )

    # -------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]
