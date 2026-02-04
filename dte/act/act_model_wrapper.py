# core/act/act_model_wrapper.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from dte.interfaces import (
    ACTModelProtocol,
    CoreProtocol,
    EncoderProtocol,
    MemoryInitProtocol,
    MemoryProtocol,
    OutputProtocol,
)
from dte.utils.distributed import DistContext


class ACTModelWrapper(nn.Module):
    """
    Pure wrapper around the base model components.
    - forward() is TRAIN entrypoint (what DDP wraps).
    - forward_eval()/forward_once() are explicit non-DDP APIs.
    - unroll_* and leaf ops are the ONLY intended torch.compile targets.

    Also supports serializing the first call(s) into compiled paths across ranks
    to avoid multi-rank Triton/Inductor JIT/cache races.
    """

    def __init__(
        self,
        base_model: ACTModelProtocol,
        *,
        train_unroll_steps: int,
        eval_unroll_steps: int,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.encoder: EncoderProtocol = base_model.encoder
        self.memory_init: MemoryInitProtocol = base_model.memory_init
        self.core: CoreProtocol = base_model.core

        if train_unroll_steps < 1:
            raise ValueError(f"train_unroll_steps must be >= 1 (got {train_unroll_steps})")
        if eval_unroll_steps < 1:
            raise ValueError(f"eval_unroll_steps must be >= 1 (got {eval_unroll_steps})")

        self.train_unroll_steps = int(train_unroll_steps)
        self.eval_unroll_steps = int(eval_unroll_steps)

        # ---- DDP first-call serialization plumbing ----
        self._dist_ctx: DistContext | None = None
        self._serialize_first_calls: bool = False
        self._first_done: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # DDP JIT race guard configuration
    # ------------------------------------------------------------------
    def set_first_call_serialization(self, *, dist_ctx: DistContext, enabled: bool) -> None:
        """
        If enabled under DDP, serializes the first invocation of selected entrypoints
        (train/eval/once) sequentially across ranks using dist_ctx.barrier(backend="gloo").
        
        this avoid compilation problem when compiling on several GPU where some node are deficients

        This is intentionally outside torch.compile targets.
        """
        self._dist_ctx = dist_ctx
        self._serialize_first_calls = bool(enabled)
        self._first_done.clear()

    def _run_first_call_serialized(self, key: str, fn, *args, **kwargs):
        """
        Serialize first invocation for `key` across ranks sequentially:
          rank 0 runs -> barrier -> rank 1 runs -> barrier -> ...
        """
        
        if not self._serialize_first_calls:
            return fn(*args, **kwargs)

        if self._first_done.get(key, False):
            return fn(*args, **kwargs)

        self._first_done[key] = True

        dist_ctx = self._dist_ctx
        if dist_ctx is None or not dist_ctx.distributed_ready or dist_ctx.world_size <= 1:
            return fn(*args, **kwargs)

        out = None
        for r in range(dist_ctx.world_size):
            dist_ctx.barrier(backend="gloo")
            if dist_ctx.rank == r:
                print(
                    f"[jit-serialize] compiling {key} on rank={dist_ctx.rank} (world_size = {dist_ctx.world_size}) "
                    f"local_rank={dist_ctx.local_rank} dev={dist_ctx.cuda_device} host={dist_ctx.host} pid={dist_ctx.pid}",
                    flush=True,
                )
                out = fn(*args, **kwargs)
            dist_ctx.barrier(backend="gloo")


        return out

    # -----------------------
    # DDP entrypoint (TRAIN)
    # This function should *not* be compiled
    # -----------------------
    @torch._dynamo.disable  # keep Python control flow eager even if someone compiles parent modules
    def forward(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        return self._run_first_call_serialized("train_forward", self.unroll_train, memory, inputs)

    # -----------------------
    # Explicit non-train APIs
    # -----------------------
    @torch.no_grad()
    @torch._dynamo.disable
    def forward_eval(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        return self._run_first_call_serialized("eval_forward", self.unroll_eval, memory, inputs)

    @torch._dynamo.disable
    def forward_once(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        return self._run_first_call_serialized("once_forward", self.step_once, memory, inputs)

    # ------------------------------------------------------------------
    # Memory API (not part of compiled path)
    # ------------------------------------------------------------------
    def init_memory(self, inputs: torch.Tensor) -> MemoryProtocol:
        return self.memory_init.init_memory(inputs)

    def reset_memory(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> MemoryProtocol:
        return self.memory_init.reset_memory(memory, inputs, reset_mask)

    # ------------------------------------------------------------------
    # Leaf ops (compile targets)
    # ------------------------------------------------------------------
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def step_encoded(
        self,
        memory: MemoryProtocol,
        encoded: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        return self.core.step(memory, encoded)

    def step_once(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        encoded = self.encode(inputs)
        return self.step_encoded(memory, encoded)

    # ------------------------------------------------------------------
    # Static-loop unrolls (compile targets)
    # ------------------------------------------------------------------
    def unroll_train(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        encoded = self.encode(inputs)

        out: OutputProtocol | None = None
        for _ in range(self.train_unroll_steps):
            memory, out = self.step_encoded(memory, encoded)

        return memory, out  # type: ignore[return-value]

    @torch.no_grad()
    def unroll_eval(
        self,
        memory: MemoryProtocol,
        inputs: torch.Tensor,
    ) -> Tuple[MemoryProtocol, OutputProtocol]:
        encoded = self.encode(inputs)

        out: OutputProtocol | None = None
        for _ in range(self.eval_unroll_steps):
            memory, out = self.step_encoded(memory, encoded)

        return memory, out  # type: ignore[return-value]
