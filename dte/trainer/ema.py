# core/trainer/ema.py
import contextlib
import torch
import torch.nn as nn


def _unwrap(module: nn.Module) -> nn.Module:
    # Works for nn.DataParallel and DistributedDataParallel (and most wrappers)
    return module.module if hasattr(module, "module") else module


class EMAHelper:
    def __init__(self, mu: float = 0.999):
        self.mu = float(mu)
        self.shadow: dict[str, torch.Tensor] = {}  # name -> tensor

    # ----------------------------------------------
    # Register params into EMA buffer
    # ----------------------------------------------
    def register(self, module: nn.Module) -> None:
        module = _unwrap(module)
        with torch.no_grad():
            for name, p in module.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone()

    # ----------------------------------------------
    # Update EMA shadow parameters
    # shadow = mu * shadow + (1-mu) * param
    # Robust to missing keys (resume / wrapper name drift / new params)
    # ----------------------------------------------
    def update(self, module: nn.Module, mu: float | None = None) -> None:
        module = _unwrap(module)
        mu = self.mu if mu is None else float(mu)

        with torch.no_grad():
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue

                buf = self.shadow.get(name)
                if buf is None:
                    # Key mismatch (common on resume). Initialize instead of crashing.
                    self.shadow[name] = p.detach().clone()
                    continue

                buf.mul_(mu).add_(p.detach(), alpha=1.0 - mu)

    # ----------------------------------------------
    # Apply EMA to a module IN PLACE (overwrite params)
    # strict=True: raise if any key missing
    # strict=False: skip missing keys
    # ----------------------------------------------
    def ema(self, module: nn.Module, strict: bool = True) -> None:
        module = _unwrap(module)
        missing = []

        with torch.no_grad():
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                buf = self.shadow.get(name)
                if buf is None:
                    missing.append(name)
                    continue
                p.data.copy_(buf)

        if strict and missing:
            raise KeyError(f"EMA missing {len(missing)} keys (e.g. {missing[0]})")

    # ----------------------------------------------
    # In-place SWAP: temporarily use EMA weights
    # strict=True: raise if any key missing
    # strict=False: only swap keys present in EMA
    # ----------------------------------------------
    @contextlib.contextmanager
    def swap(self, module: nn.Module, strict: bool = True):
        module = _unwrap(module)

        backup: dict[str, torch.Tensor] = {}
        missing = []

        with torch.no_grad():
            # Store original params
            for name, p in module.named_parameters():
                if p.requires_grad:
                    backup[name] = p.detach().clone()

            # Load EMA params in-place
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                buf = self.shadow.get(name)
                if buf is None:
                    missing.append(name)
                    continue
                p.data.copy_(buf)

        if strict and missing:
            # Restore before raising
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if p.requires_grad and name in backup:
                        p.data.copy_(backup[name])
            raise KeyError(f"EMA missing {len(missing)} keys (e.g. {missing[0]})")

        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if p.requires_grad and name in backup:
                        p.data.copy_(backup[name])

    # ----------------------------------------------
    # Save / load state dict for checkpoints
    # ----------------------------------------------
    def state_dict(self):
        return {name: buf.clone() for name, buf in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}
