from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from dte.trainer.train_state import TrainingState
from dte.utils.distributed import DistContext
from dte.trainer.optimization_helper import OptimizationHelper
from dte.trainer.metric_helper import MetricHelper
from dte.trainer.ema import EMAHelper
from dte.act.act_container import ACTContainerManager
from dte.interfaces import LossProtocol, HaltingController


@dataclass
class TrainingContext:

    train_state: TrainingState
    dist: DistContext

    optimizer: OptimizationHelper
    metric_helper: MetricHelper
    act_manager: ACTContainerManager
    criterion: LossProtocol
    halting_ctrl: HaltingController

    ema: Optional[EMAHelper] = None
    checkpoint_manager: Any = None