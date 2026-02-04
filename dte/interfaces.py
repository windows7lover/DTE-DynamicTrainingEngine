from typing import Protocol, Tuple, Any, overload
from typing_extensions import Self
import numpy as np
import torch


# =========================================================
# Online dataset generator Protocol
# =========================================================
class GeneratorProtocol(Protocol):
    """
    A generator may implement:

    1. Per-sample:
         __call__(rng) -> (inp[L], lbl[L])
         __call__(rng) -> (inp[L], lbl[L], sample_id)

    2. Batched:
         __call__(rng, batch_size) -> (inp[B,L], lbl[B,L])
         __call__(rng, batch_size) -> (inp[B,L], lbl[B,L], sample_id[B])

    OnlineDataset handles all forms transparently.
    """

    @overload
    def __call__(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def __call__(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def __call__(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    @overload
    def __call__(
        self,
        rng: np.random.Generator,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def __call__(self, *args, **kwargs):
        ...

# =========================================================
# Encoder / Core / MemoryInit Protocols
# =========================================================

class EncoderProtocol(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw input IDs or tokens into continuous embeddings.
        """
        ...

class MemoryProtocol(Protocol):
    def to(self: Self, device: torch.device) -> Self:
        """
        Must return a new or in-place-updated instance of the memory state
        moved to the specified device.
        """
        ...
        
class MemoryInitProtocol(Protocol):
    def init_memory(
        self,
        inputs: torch.Tensor,
    ) -> "MemoryProtocol":
        """
        Initialize ACT memory for a full batch.
        Inputs may be used to condition the initial memory.
        """
        ...

    def reset_memory(
        self,
        memory: "MemoryProtocol",
        inputs: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> "MemoryProtocol":
        """
        Reset only masked rows of the memory.
        Inputs may be used to condition the reset.
        """
        ...


class CoreProtocol(Protocol):
    def step(
        self,
        memory: "MemoryProtocol",
        encoded_inputs: torch.Tensor,
    ) -> tuple["MemoryProtocol", "OutputProtocol"]:
        """
        One reasoning step given encoded inputs and the current memory.
        """
        ...
        
class ACTModelProtocol(Protocol):
    """
    A model usable with ACT must expose exactly three components:

        encoder:      maps input tokens -> embeddings
        core:         recurrent reasoning block (step function)
        memory_init:  memory initialization + reset logic

    No ACT logic (forward/step/init_memory/reset_memory) should be
    implemented by the model itself - that belongs to the ACT wrapper.
    """
    encoder: EncoderProtocol
    core: CoreProtocol
    memory_init: MemoryInitProtocol


# =========================================================
# Output Protocol
# =========================================================
class OutputProtocol(Protocol):
    def reset(self: Self, reset_mask: torch.Tensor | None) -> None:
        """
        Reset this output in-place.
        - If reset_mask is None -> full reset
        - Otherwise -> reset only masked rows.
        """
        ...

    def to(self: Self, device: torch.device) -> Self:
        """
        Must return self with all internal tensors moved to the given device.
        """
        ...


# ---------------------------------------------------------
# Loss function interface (for criterion)
# ---------------------------------------------------------
class LossProtocol(Protocol):
    def __call__(
        self,
        output: Any,
        model_memory: Any,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute LM + halting loss.
        Must return (loss, info_dict).
        """
        ...


# ---------------------------------------------------------
# Halting policy interface (for HaltingController and HaltingState)
# ---------------------------------------------------------
class HaltingStateProtocol(Protocol):
    def to(self: Self, device: torch.device) -> Self:
        """
        Must return a version of the halting state on the given device.
        """
        ...


class HaltingController(Protocol):
    def init_halting_state(self, batch_size: int, device: torch.device) -> HaltingStateProtocol:
        ...

    def reset_halting_state(
        self,
        state: HaltingStateProtocol,
        reset_mask: torch.Tensor,
    ) -> HaltingStateProtocol:
        ...

    def get_halted_mask(self, state: HaltingStateProtocol) -> torch.Tensor:
        ...

    def update_halting_state(
        self,
        halting_state: HaltingStateProtocol,
        model_memory: Any,
        outputs: Any,
        loss_info: dict,
        training: bool,
    ) -> HaltingStateProtocol:
        ...


# ---------------------------------------------------------
# Metrics function interface
# ---------------------------------------------------------
class MetricsFn(Protocol):
    def __call__(
        self,
        model_memory: Any,
        outputs: Any,
        labels: torch.Tensor,
        loss_info: dict,
        phase: str,
        halted_mask: torch.Tensor,
        halting_state: Any | None,
    ):
        """
        Should return MetricRecord.
        """
        ...
