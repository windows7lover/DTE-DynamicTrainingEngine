from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from models.trm import TRMMemory
import torch

@dataclass
class HaltingState:
    halted: torch.Tensor         # bool[B]
    min_halt_steps: torch.Tensor # int[B]

    def to(self, device: torch.device) -> "HaltingState":
        return HaltingState(
            halted=self.halted.to(device),
            min_halt_steps=self.min_halt_steps.to(device),
        )

class HaltingController:
    """
    Halting policy for ACT.

    Stores:
      - max_steps
      - exploration parameters
      - per-sample minimum halt step (for exploration)
      - dataset metadata (kept for possible future use)
    """

    def __init__(
        self,
        max_steps: int,
        exploration_enabled: bool = True,
        exploration_prob: float = 0.0,
        device: str = "cpu",
        metadata: Optional[DatasetMetadata] = None,
    ):
        self.max_steps = max_steps
        self.exploration_enabled = exploration_enabled
        self.exploration_prob = exploration_prob

        self.metadata = metadata
        
    def init_halting_state(self, batch_size: int, device: torch.device) -> HaltingState:
        """
        Initialize halting halting_state for a new container.
        By default:
          - all samples start as halted=True (to replace empty data)
          - min_halt_steps = 0
        """
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        min_halt_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return HaltingState(halted=halted, min_halt_steps=min_halt_steps)


    # -------------------------------------------------------------
    # Reset + assign for newly replaced samples
    # -------------------------------------------------------------
    def reset_halting_state(self, halting_state: HaltingState, reset_mask: torch.Tensor) -> HaltingState:
        """
        Reset halting halting_state for rows in reset_mask.
        - halted -> False
        - min_halt_steps -> 0, then re-sampled for exploration if enabled
        """
        idx = reset_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return halting_state

        halted = halting_state.halted.clone()
        min_halt_steps = halting_state.min_halt_steps.clone()

        # Clear halted + constraints
        halted.index_fill_(0, idx, False)
        min_halt_steps.index_fill_(0, idx, 0)

        # Exploration constraints
        if self.exploration_enabled and self.exploration_prob > 0:
            explore_mask = torch.rand(idx.numel(), device=idx.device) < self.exploration_prob
            if explore_mask.any():
                chosen = idx[explore_mask]
                min_halt_steps[chosen] = torch.randint(
                    low=2,
                    high=self.max_steps + 1,
                    size=(explore_mask.sum(),),
                    device=idx.device,
                    dtype=torch.int32,
                )

        return HaltingState(halted=halted, min_halt_steps=min_halt_steps)


    # -------------------------------------------------------------
    # Halting policy
    # -------------------------------------------------------------
    def update_halting_state(
        self,
        halting_state: HaltingState,
        model_memory: TRMMemory,
        outputs,
        loss_info: dict,
        *,
        training: bool,
    ) -> HaltingState:
        """
        One halting update step:
          - reads model outputs + loss_info + steps
          - returns a NEW HaltingState
        """
        assert outputs is not None

        reached_limit = model_memory.steps >= self.max_steps

        # In eval: only hard cap at max_steps
        if not training:
            halted = reached_limit
            return HaltingState(halted=halted, min_halt_steps=halting_state.min_halt_steps)

        # seq_correct precomputed by the criterion (boolean [B])
        seq_correct = loss_info["halt/seq_correct"]
        if seq_correct.dtype != torch.bool:
            seq_correct = seq_correct > 0.5

        model_halt = outputs.q_halt > 0
        pass_exploration = (model_memory.steps >= halting_state.min_halt_steps)

        halted = reached_limit | (model_halt & seq_correct & pass_exploration)
        return HaltingState(halted=halted, min_halt_steps=halting_state.min_halt_steps)

    def get_halted_mask(self, halting_state: HaltingState) -> torch.Tensor:
        return halting_state.halted
