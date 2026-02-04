# core/act/act_container.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from dte.data.batch import Batch
from dte.interfaces import MemoryProtocol, OutputProtocol, HaltingStateProtocol


class DataStreamProtocol(Protocol):
    def next_n_samples(self, n: int) -> Batch: ...


@dataclass
class ACTContainer:
    inputs: torch.Tensor
    labels: torch.Tensor
    model_memory: MemoryProtocol | None
    outputs: OutputProtocol | None
    halting_state: HaltingStateProtocol | None
    sample_id: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])


class ACTContainerManager:
    """
    Orchestration only:
      - device normalization
      - row replacement/reset
      - calling wrapper's forward
    """

    def __init__(
        self,
        *,
        halting_ctrl,
        init_memory_fn,
        reset_memory_fn,
        model_wrapper,  # ACTModelWrapper or DDP(ACTModelWrapper)
        device: torch.device,
        batch_size: int,  # per-rank batch size
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {batch_size})")

        self.halting_ctrl = halting_ctrl
        self.init_memory_fn = init_memory_fn
        self.reset_memory_fn = reset_memory_fn
        self.model_wrapper = model_wrapper
        self.device = device
        self.batch_size = int(batch_size)

    def move_container_to_device(self, container: ACTContainer) -> ACTContainer:
        dev = self.device

        def _mv(t: torch.Tensor) -> torch.Tensor:
            return t if t.device == dev else t.to(dev, non_blocking=True)

        container.inputs = _mv(container.inputs)
        container.labels = _mv(container.labels)

        if container.sample_id is not None:
            container.sample_id = _mv(container.sample_id)
        if container.model_memory is not None:
            container.model_memory = container.model_memory.to(dev)  # custom objects should implement fast .to
        if container.outputs is not None:
            container.outputs = container.outputs.to(dev)
        if container.halting_state is not None:
            container.halting_state = container.halting_state.to(dev)

        return container

    def allocate_empty_container_from_batch(self, batch: Batch) -> ACTContainer:
        batch = batch.to(self.device)
        B = int(batch.inputs.shape[0])

        empty_inputs = torch.empty_like(batch.inputs)
        empty_labels = torch.empty_like(batch.labels)
        empty_sample_id = torch.empty_like(batch.sample_id) if batch.sample_id is not None else None

        memory = self.init_memory_fn(batch.inputs)
        halting = self.halting_ctrl.init_halting_state(batch_size=B, device=self.device)

        return ACTContainer(
            inputs=empty_inputs,
            labels=empty_labels,
            model_memory=memory.to(self.device),
            outputs=None,
            halting_state=halting.to(self.device),
            sample_id=empty_sample_id,
        )

    def ensure_container_and_refresh_halted(
        self,
        *,
        container: ACTContainer | None,
        stream: DataStreamProtocol,
    ) -> ACTContainer:
        if container is None:
            new_samples = stream.next_n_samples(self.batch_size)
            container = self.allocate_empty_container_from_batch(new_samples)

            # keep both views consistent
            reset_mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
            reset_mask_cpu = torch.ones(self.batch_size, device="cpu", dtype=torch.bool)

            self.replace_and_reset_rows(container=container, new_samples=new_samples, reset_mask=reset_mask)
            return container

        hs = container.halting_state
        assert hs is not None, "container.halting_state must not be None"

        reset_mask = self.halting_ctrl.get_halted_mask(hs)
        reset_mask_cpu = reset_mask.detach().to("cpu", non_blocking=False)

        if not reset_mask_cpu.any():
            return container

        n_reset = int(reset_mask_cpu.sum())
        new_samples = stream.next_n_samples(n_reset)

        self.replace_and_reset_rows(container=container, new_samples=new_samples, reset_mask=reset_mask)
        return container


    def replace_and_reset_rows(
        self,
        container: ACTContainer,
        new_samples: Batch,
        reset_mask: torch.Tensor,  # expected CUDA bool[B] for indexing
    ) -> ACTContainer:
        if not torch.any(reset_mask):
            return container

        new_samples = new_samples.to(self.device)
        new_inputs = new_samples.inputs
        new_labels = new_samples.labels
        
        container.inputs[reset_mask] = new_inputs
        container.labels[reset_mask] = new_labels

        if container.sample_id is not None and getattr(new_samples, "sample_id", None) is not None:
            container.sample_id[reset_mask] = new_samples.sample_id

        if container.model_memory is not None:
            container.model_memory = self.reset_memory_fn(container.model_memory, container.inputs, reset_mask)

        if container.outputs is not None:
            container.outputs.reset(reset_mask)

        if container.halting_state is not None:
            container.halting_state = self.halting_ctrl.reset_halting_state(container.halting_state, reset_mask)

        return self.move_container_to_device(container)


    # -------------------------------------------------------------------------
    # Compute entrypoint
    # -------------------------------------------------------------------------
    def run(self, container: ACTContainer, mode: str):
        container = self.move_container_to_device(container)
        mem = container.model_memory
        x = container.inputs
        assert mem is not None

        mw = self.model_wrapper
        inner = mw.module if hasattr(mw, "module") else mw

        if mode == "train":
            mem, out = mw(mem, x)                 # DDP forward
            steps = inner.train_unroll_steps
        elif mode == "eval":
            mem, out = inner.forward_eval(mem, x) # no_grad, no DDP needed
            steps = inner.eval_unroll_steps
        elif mode == "once":
            mem, out = inner.forward_once(mem, x)
            steps = 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        container.model_memory = mem
        container.outputs = out
        return container, steps
