# core/utils/distributed.py
from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from typing import Optional

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    cpu_group: Optional[dist.ProcessGroup]  # None iff world_size == 1

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def host(self) -> str:
        return socket.gethostname()

    @property
    def pid(self) -> int:
        return os.getpid()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    @property
    def cuda_device(self):
        return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    @property
    def distributed_ready(self) -> bool:
        return self.world_size > 1 and dist.is_available() and dist.is_initialized()

    def barrier(self, *, backend: str = "gloo") -> None:
        """
        backend:
          - "gloo": barrier on cpu_group (preferred)
          - "nccl": barrier on default group
        """
        if not self.distributed_ready:
            return

        if backend == "gloo":
            if self.cpu_group is None:
                raise RuntimeError("cpu_group is None but distributed_ready=True (setup_distributed bug).")
            dist.barrier(group=self.cpu_group)
            return

        if backend == "nccl":
            dist.barrier()
            return

        raise ValueError(f"Unknown backend='{backend}' (expected 'gloo' or 'nccl').")

    def dbg(self, msg: str) -> None:
        print(
            f"[host={self.host} pid={self.pid} rank={self.rank}/{self.world_size} "
            f"local_rank={self.local_rank} dev={self.cuda_device}] {msg}",
            flush=True,
        )


def setup_distributed(*, backend: str = "nccl") -> DistContext:
    """
    Initializes default process group if WORLD_SIZE>1 (torchrun env://).
    Always pins CUDA device from LOCAL_RANK before init.
    Also creates a CPU/Gloo group spanning all ranks for safe barriers/debug.
    """
    world_size_env = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)

    # Pin device early (safe even for world_size==1)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size_env <= 1:
        return DistContext(rank=0, world_size=1, local_rank=local_rank, cpu_group=None)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1.")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a CPU/Gloo group spanning all ranks
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank, cpu_group=cpu_group)


def teardown_distributed(dist_ctx: Optional[DistContext]) -> None:
    """
    Call on ALL ranks (best in try/finally).
    Destroys custom groups first, then the default group.
    """
    if dist_ctx is None:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return

    # Best-effort sync to avoid destroying during collectives
    try:
        dist_ctx.barrier(backend="gloo")
    except Exception:
        pass

    if dist_ctx.cpu_group is not None:
        try:
            dist.destroy_process_group(dist_ctx.cpu_group)
        except Exception:
            pass

    try:
        dist.destroy_process_group()
    except Exception:
        pass


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def master_print(dist_ctx: DistContext, *args, **kwargs) -> None:
    if dist_ctx.is_master:
        print(*args, **kwargs)
        
def ddp_check_grad_synced(ddp_model, dist_ctx, name_hint=None):
    m = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    # pick a deterministic param (first named parameter, or a specific name)
    named = list(m.named_parameters())
    if name_hint is not None:
        for n, p in named:
            if name_hint in n:
                target_name, target_p = n, p
                break
        else:
            target_name, target_p = named[0]
    else:
        target_name, target_p = named[0]

    g = target_p.grad
    if g is None:
        fp = torch.zeros(3, device=target_p.device)
    else:
        gf = g.detach().float()
        fp = torch.tensor([gf.mean(), gf.abs().mean(), gf.norm()], device=gf.device)

    gathered = [torch.empty_like(fp) for _ in range(dist_ctx.world_size)]
    dist.all_gather(gathered, fp)

    if dist_ctx.is_master:
        vals = torch.stack(gathered).cpu()
        print(f"[ddp-check] param={target_name} fps={vals.tolist()}", flush=True)
        print(f"[ddp-check] max diff={(vals - vals[0]).abs().max().item():.3e}", flush=True)
