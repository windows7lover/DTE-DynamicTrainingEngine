from __future__ import annotations
from dataclasses import dataclass, field

from collections import deque
from typing import Dict, Optional, Tuple, Any

import torch
import torch.distributed as dist
from dte.utils.distributed import DistContext


class MetricHelper:
    """
    Structured, agnostic metric processor.

    scalar_modes[name] =
        ("sum",)
        ("mean", denom_key)
        ("ignore",)

    denom_key:
        - any scalar name inside rec.scalars
          (helper does NOT know batch_size or world_size)
    """

    def __init__(self, dist_ctx: DistContext, window_size: int = 1):
        self.dist_ctx = dist_ctx
        self.world_size = dist_ctx.world_size
        self.is_master = dist_ctx.is_master
        self.window_size = window_size

        # rolling buffers for smoothing vector metrics
        self.smooth_buffers: Dict[str, deque] = {}

    # -------------------------------------------------------------
    # reduce scalars across DDP
    # -------------------------------------------------------------
    def _reduce_scalars(self, rec: "MetricRecord") -> Dict[str, float]:
        raw = rec.to_raw_dict()
        local_keys = sorted(raw.keys())

        # All ranks must participate in the same collectives in the same order,
        # even if a rank has no metrics this step.
        keys = local_keys
        if self.world_size > 1 and dist.is_initialized():
            all_keys = [None] * self.world_size
            dist.all_gather_object(all_keys, local_keys)

            if len({tuple(k) for k in all_keys}) != 1:
                print("Metric key mismatch across ranks:", all_keys, flush=True)
                raise RuntimeError("Metric keys diverged")

            keys = all_keys[0]  # agreed key order across ranks

        # No metrics anywhere: nothing to reduce
        if not keys:
            return {}

        # Build value vector in agreed key order (shape must match across ranks)
        values = torch.stack([raw[k] for k in keys])

        if self.world_size > 1 and dist.is_initialized():
            device = self.dist_ctx.device
            values = values.to(device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values = values.cpu()

        return {k: float(values[i]) for i, k in enumerate(keys)}



    # -------------------------------------------------------------
    # normalize scalars using structured modes
    # -------------------------------------------------------------
    def _normalize_scalars(
        self,
        rec: MetricRecord,
        reduced: Dict[str, float],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for k, spec in rec.scalar_modes.items():
            mode = spec[0]

            if mode == "ignore":
                continue

            if mode == "sum":
                if k in reduced:
                    out[k] = reduced[k]
                continue

            if mode == "mean":
                denom_key = spec[1]
                denom = reduced.get(denom_key, None)
                if denom is None or denom == 0:
                    continue
                if k in reduced:
                    out[k] = reduced[k] / denom
                continue

        return out

    # -------------------------------------------------------------
    # vector metrics
    # -------------------------------------------------------------
    def _process_vectors(self, rec: MetricRecord) -> Dict[str, float]:
        out: Dict[str, float] = {}

        for name, vec in rec.vectors.items():
            mode = rec.vector_modes.get(name, "ignore")
            if mode == "ignore":
                continue

            vec = vec.to(torch.float32)

            if mode == "raw":
                out[name] = float(vec.mean().item())
                continue

            if mode == "smooth":
                buf = self.smooth_buffers.setdefault(
                    name, deque(maxlen=self.window_size)
                )
                for x in vec.reshape(-1).tolist():
                    buf.append(float(x))

                val = sum(buf) / len(buf) if len(buf) else 0.0
                out[f"{name}_smoothed"] = val
                continue

        return out

    # -------------------------------------------------------------
    # unified metric processing (train/eval)
    # -------------------------------------------------------------
    def process_metrics(
        self,
        rec: MetricRecord,
        rank: int, # in case of distributed metrics
        extra_scalars: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:

        reduced = self._reduce_scalars(rec)

        if not self.is_master:
            return None

        normalized = self._normalize_scalars(rec, reduced)
        vectors = self._process_vectors(rec)

        prefix = f"{rec.phase}/"
        out: Dict[str, float] = {}

        for k, v in normalized.items():
            out[prefix + k] = v

        for k, v in vectors.items():
            out[prefix + k] = v

        if extra_scalars:
            out.update(extra_scalars)

        return out

# ------------------------------------------------------------------
# MetricRecord
# ------------------------------------------------------------------
@dataclass
class MetricRecord:
    """
    scalars:         raw scalar sums (to reduce by SUM across DDP)
    scalar_modes:    dict name → ("sum") OR ("mean", denom_key)

    vectors:         vector-valued metrics for smoothing
    vector_modes:    dict name → "smooth" | "raw" | "ignore"

    phase:           train / eval / custom, used for prefixing in MetricHelper
    """
    scalars: Dict[str, torch.Tensor] = field(default_factory=dict)
    scalar_modes: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)

    vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    vector_modes: Dict[str, str] = field(default_factory=dict)

    phase: str = "train"

    def to_raw_dict(self):
        return dict(self.scalars)

# -------------------------------------------------------------
# accumulator
# -------------------------------------------------------------
class MetricAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.storage: Dict[str, float] = {}
        self.count = 0

    def add(self, metrics: Optional[Dict[str, float]]):
        if metrics is None:
            return
        for k, v in metrics.items():
            self.storage[k] = self.storage.get(k, 0.0) + float(v)
        self.count += 1

    def finalize(self):
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.storage.items()}
