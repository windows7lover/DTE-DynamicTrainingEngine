# core/trainer/checkpoint.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from dte.trainer.training_context import TrainingContext
from dte.utils.distributed import DistContext, unwrap


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _get_rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _set_rng_state(state) -> None:
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


@dataclass
class CheckpointManager:
    ctx: TrainingContext
    base_path: str

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    def _train_dir(self) -> str:
        return os.path.join(self.base_path, "train")

    def _model_dir(self) -> str:
        return os.path.join(self.base_path, "model")

    def _ema_dir(self) -> str:
        return os.path.join(self.base_path, "ema")

    # -------------------------------------------------
    # SAVE (master-only)
    # -------------------------------------------------
    def save(self) -> None:
        dist_ctx: DistContext = self.ctx.dist
        if not dist_ctx.is_master:
            return

        os.makedirs(self._train_dir(), exist_ok=True)
        os.makedirs(self._model_dir(), exist_ok=True)
        os.makedirs(self._ema_dir(), exist_ok=True)

        ts = self.ctx.train_state
        model = unwrap(ts.model)

        step = ts.step
        fname = f"step_{step:08d}.pt"

        train_ckpt = {
            "version": 1,
            "model": model.state_dict(),
            "optimizer": (
                self.ctx.optimizer.optimizer.state_dict()
                if self.ctx.optimizer.optimizer is not None
                else None
            ),
            "ema": self.ctx.ema.state_dict() if self.ctx.ema is not None else None,
            "train_state": {"step": int(ts.step), "epoch": int(ts.epoch)},
            "rng": _get_rng_state(),
        }

        train_path = os.path.join(self._train_dir(), fname)
        _atomic_save(train_ckpt, train_path)
        _atomic_save(train_ckpt, os.path.join(self._train_dir(), "last.pt"))

        _atomic_save({"version": 1, "model": train_ckpt["model"]},
                     os.path.join(self._model_dir(), "last.pt"))

        if self.ctx.ema is not None:
            _atomic_save({"version": 1, "ema": train_ckpt["ema"]},
                         os.path.join(self._ema_dir(), "last.pt"))

    # -------------------------------------------------
    # LOAD (all ranks)
    # -------------------------------------------------
    def load(self, path: str) -> None:
        dist_ctx: DistContext = self.ctx.dist
        ts = self.ctx.train_state

        if dist_ctx.is_master:
            print(f"[resume] loading checkpoint from: {path}", flush=True)

        # Each rank loads from shared filesystem (standard torchrun pattern)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # -------------------------
        # Model
        # -------------------------
        model = unwrap(ts.model)
        model.load_state_dict(ckpt["model"], strict=True)

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{dist_ctx.local_rank}")
            model.to(device)
            torch.cuda.synchronize()

        ts.model = model

        # -------------------------
        # Optimizer
        # -------------------------
        opt_sd = ckpt.get("optimizer")
        if opt_sd is not None:
            if self.ctx.optimizer.optimizer is None:
                self.ctx.optimizer.optimizer = self.ctx.optimizer._build_optimizer(ts.model)
            opt = self.ctx.optimizer.optimizer
            opt.load_state_dict(opt_sd)

            # Ensure optimizer state tensors are on same device as params (common gotcha)
            if torch.cuda.is_available():
                dev = next(model.parameters()).device
                for st in opt.state.values():
                    for k, v in list(st.items()):
                        if torch.is_tensor(v) and v.device != dev:
                            st[k] = v.to(dev, non_blocking=True)
                torch.cuda.synchronize()

        # -------------------------
        # EMA
        # -------------------------
        ema_sd = ckpt.get("ema")
        if self.ctx.ema is not None and ema_sd is not None:
            self.ctx.ema.load_state_dict(ema_sd)
            if torch.cuda.is_available():
                dev = next(model.parameters()).device
                # assumes your EMA implementation keeps tensors in .shadow
                for k, v in self.ctx.ema.shadow.items():
                    self.ctx.ema.shadow[k] = v.to(dev, non_blocking=True)
                torch.cuda.synchronize()

        # -------------------------
        # Train state + RNG + derived
        # -------------------------
        ts.step = int(ckpt["train_state"]["step"])
        ts.epoch = int(ckpt["train_state"]["epoch"])
        _set_rng_state(ckpt["rng"])

        ts.container = None

        # Single sync point (safe under torchrun; uses default PG backend)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
