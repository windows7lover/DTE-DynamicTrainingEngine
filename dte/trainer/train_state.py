from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from dte.utils.config_loader import PretrainConfig
from dte.interfaces import MetricsFn


@dataclass
class TrainingState:
    config: PretrainConfig
    rank: int
    world_size: int
    max_training_steps_per_epoch: int
    max_eval_steps_per_epoch: int
    max_epoch: int
    data_module: Any
    device: torch.device

    # runtime objects
    model: nn.Module | None = None
    metrics_fn: MetricsFn | None = None


    # training progress
    step: int = field(init=False, default=0)
    epoch: int = field(init=False, default=0)
    
    @property
    def total_training_steps(self) -> int:
        return self.max_epoch * self.max_training_steps_per_epoch

