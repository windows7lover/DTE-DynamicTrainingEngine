# core/act/act_model_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dte.act.act_model_wrapper import ACTModelWrapper


@dataclass(frozen=True)
class AssembledACTModel:
    train_model: nn.Module          # ACTModelWrapper or DDP(ACTModelWrapper)
    inner_wrapper: ACTModelWrapper  # always unwrapped ACTModelWrapper


class ACTModelFactory:
    """
    Single place that:
      base_model -> ACTModelWrapper -> (optional) DDP wrapper

    No compilation here; compilation is a separate, explicit step that can be run post-resume
    """

    @staticmethod
    def build(
        *,
        base_model: Any,
        device: torch.device,
        dist: Any,
        train_unroll_steps: int,
        eval_unroll_steps: int,
        ddp_find_unused_parameters: bool = False,
        ddp_static_graph: bool = False,
    ) -> AssembledACTModel:
        wrapper = ACTModelWrapper(
            base_model,
            train_unroll_steps=train_unroll_steps,
            eval_unroll_steps=eval_unroll_steps,
        ).to(device)
        
        is_dist_cuda_enabled = (dist.world_size > 1 and device.type == "cuda" and device.index is not None)
        
        wrapper.set_first_call_serialization(
            dist_ctx=dist,
            enabled=is_dist_cuda_enabled,
        )

        train_model: nn.Module = wrapper

        if is_dist_cuda_enabled:
            train_model = DDP(
                wrapper,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=ddp_find_unused_parameters,
                static_graph=ddp_static_graph,
                gradient_as_bucket_view=True,
            )

        inner = train_model.module if hasattr(train_model, "module") else train_model
        assert isinstance(inner, ACTModelWrapper), f"Expected ACTModelWrapper, got {type(inner)}"
        return AssembledACTModel(train_model=train_model, inner_wrapper=inner)


class ACTModelCompiler:
    """
    Owns ONLY torch.compile policy for ACT wrapper functions.
    """

    @staticmethod
    def compile(*, wrapper, torch_compile_cfg, dist) -> None:
        if torch_compile_cfg is None or not torch_compile_cfg.enabled:
            return

        tcfg = torch_compile_cfg

        def compile_with_config(fn, default_cfg, local_cfg, name: str):
            enabled = getattr(local_cfg, "enabled", None)
            if enabled is None:
                enabled = getattr(default_cfg, "enabled", None)
            if enabled is None:
                enabled = getattr(tcfg, "enabled", None)

            if not enabled:
                if dist.is_master:
                    print(f"[compile] {name}: disabled", flush=True)
                return fn

            kwargs = {}
            for field in ("backend", "mode", "fullgraph", "dynamic", "options"):
                val = getattr(local_cfg, field, None)
                if val is None:
                    val = getattr(default_cfg, field, None)
                if val is not None:
                    kwargs[field] = val

            if dist.is_master:
                print(f"[compile] {name}: torch.compile({kwargs})", flush=True)

            return torch.compile(fn, **kwargs)

        if dist.is_master:
            print("=== Compiling ACT functions (post-resume) ===", flush=True)

        # ------------------------------------------------------------------
        # TRAIN: compile only the largest stable unit
        # ------------------------------------------------------------------
        wrapper.unroll_train = compile_with_config(
            wrapper.unroll_train, tcfg.default, tcfg.unroll_train, "unroll_train"
        )

        # ------------------------------------------------------------------
        # EVAL: explicit semantics
        # ------------------------------------------------------------------
        if getattr(tcfg.unroll_eval, "enabled", False):
            # Compile full eval unroll if needed
            wrapper.unroll_eval = compile_with_config(
                wrapper.unroll_eval, tcfg.default, tcfg.unroll_eval, "unroll_eval"
            )
        else:
            # Default: eager eval loop + compiled leaf kernels
            wrapper.encode = compile_with_config(
                wrapper.encode, tcfg.default, tcfg.encode, "encode"
            )
            wrapper.step_encoded = compile_with_config(
                wrapper.step_encoded, tcfg.default, tcfg.step_encoded, "step_encoded"
            )

            if dist.is_master:
                print("[compile] unroll_eval: eager loop + compiled encode / step_encoded", flush=True)

            # Fail fast if disabled unexpectedly
            assert hasattr(wrapper.encode, "__wrapped__"), "encode must be compiled for eval"
            assert hasattr(wrapper.step_encoded, "__wrapped__"), "step_encoded must be compiled for eval"