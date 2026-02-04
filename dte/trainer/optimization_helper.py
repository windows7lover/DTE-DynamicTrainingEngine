from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dte.utils.distributed import DistContext
from dte.utils.optimizers_utils import compute_lr
from dte.trainer.train_state import TrainingState


class OptimizationHelper:
    """
    Centralizes optimizer + LR scheduling + distributed grad reduction.

    - Optimizer receives lr=0, scheduler sets real lr each step.
    - scheduler_cfg: base_lr, warmup, min_ratio, etc.
    - optimizer_cfg: {name: "adamw", betas, weight_decay, ...}
    """

    def __init__(
        self,
        optimizer_cfg: dict,
        scheduler_cfg: dict,
        grad_clip: float | None = None,
    ):
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.grad_clip = grad_clip

        self.optimizer: torch.optim.Optimizer | None = None  # lazy init

    # ----------------------------------------------------
    # Optimizer construction
    # ----------------------------------------------------
    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        cfg = self.optimizer_cfg
        name = cfg["name"].lower()

        if name == "adamw":
            betas = tuple(cfg.get("betas", (0.9, 0.95)))
            wd = cfg.get("weight_decay", 0.0)

            # LR intentionally omitted (scheduler overwrites)
            return torch.optim.AdamW(
                model.parameters(),
                lr=0.0,  # placeholder: scheduler sets real lr each step
                betas=betas,
                weight_decay=wd,
            )
            
        if name in {"adam_atan2", "adamatan2", "adam-atan2"}:
            
            warnings.warn(
                "Optimizer 'adam_atan2' is DEPRECATED and will be removed. "
                "It is kept only for backward compatibility and is implemented "
                "via a compatibility shim (adam-atan2-pytorch). "
                "Switch to adam_atan2_pytorch or AdamW unless you need this baseline.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        
        if name in {"adam_atan2_pytorch", "adamatan2_pytorch", "adam-atan2_pytorch"}:
            
            from adam_atan2_pytorch import AdamAtan2
            
            betas = tuple(cfg.get("betas", (0.9, 0.95)))
            wd = cfg.get("weight_decay", 0.0)

            return AdamAtan2(
                model.parameters(),
                lr=1.0,  # scheduler-controlled (1.0 is placeholder to avoid assert error)
                betas=betas,
                weight_decay=wd,
            )

        raise ValueError(f"Unknown optimizer '{name}'")

    # ----------------------------------------------------
    # Step
    # ----------------------------------------------------
    def step(self, train_state: TrainingState) -> float:
        if self.optimizer is None:
            self.optimizer = self._build_optimizer(train_state.model)

        lr = compute_lr(
            base_lr=self.scheduler_cfg["base_lr"],
            scheduler_cfg=self.scheduler_cfg,
            train_state=train_state,
        )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                train_state.model.parameters(),
                self.grad_clip,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return lr
