from __future__ import annotations
from typing import Dict, Tuple, Any

import torch
from dte.trainer.metric_helper import MetricRecord

def compute_trm_metrics(
    model_memory: Any,                    # Model internal model_memory
    outputs: Any,                         # Model output
    labels: torch.Tensor,                 # Labels
    loss_info: Dict[str, torch.Tensor],   # Criterion output
    phase: str = "train",
    halted_mask: torch.Tensor | None = None,  # [B], from halting policy
    halting_state: Any | None = None,     # e.g. HaltingState; currently unused
) -> MetricRecord:
    """
    Pure functional ACT metrics:
      - depends only on model_memory, outputs, labels, loss_info, halted_mask (+ optional halting_state)

    Args:
        model_memory: model model_memory object (must expose .steps: [B])
        outputs: model outputs object (must expose .q_halt: [B] or [B,1])
        labels: int tensor [B, L]
        loss_info: dict from criterion (must contain halt/seq_correct, halt/token_correct, loss/lm, loss/halt)
        phase: "train" or "eval"
        halted_mask: bool tensor [B] indicating which sequences are halted
        batch_size: optional explicit batch size; if None, inferred from labels.shape[0]
        halting_state: optional full halting state object (for future use)
    """

    assert halted_mask is not None, "halted_mask must be provided to compute_act_metrics"

    steps = model_memory.steps                       # [B]
    device = steps.device
    batch_size = labels.shape[0]

    # =============================================================
    # 1) EXTRACT METRICS FROM loss_info INSTEAD OF RECOMPUTING
    # =============================================================
    seq_correct = loss_info["halt/seq_correct"]       # [B]
    token_correct = loss_info["halt/token_correct"]   # [B, L]

    # token_counts = number of valid positions per sequence
    token_counts = token_correct.sum(-1)

    # Per-sequence accuracy (float vector)
    per_seq_acc = (
        token_correct.sum(-1) / token_counts.clamp(min=1)
    ).float()

    # Wrong sequences
    seq_wrong = ~seq_correct

    # q_halt correctness relative to correctness of output
    q_h = outputs.q_halt.squeeze(-1)  # [B]
    q_pred = (q_h >= 0)

    q_halt_correct_output_vec = q_pred[seq_correct].float()
    q_halt_wrong_output_vec   = (~q_pred[seq_wrong]).float()

    # valid final sequences (halted AND with at least one valid token)
    valid_final = halted_mask & (token_counts > 0)

    # =============================================================
    # Base scalars
    # =============================================================
    scalars = {
        "count": valid_final.sum(),
        "batch_size": torch.tensor(batch_size, dtype=torch.float32, device=device),
        "lm_loss": loss_info["loss/lm"],
        "q_halt_loss": loss_info["loss/halt"],
    }

    scalar_modes = {
        "count": ("ignore",),
        "batch_size": ("ignore",),
        "lm_loss": ("mean", "batch_size"),
        "q_halt_loss": ("mean", "batch_size"),
    }

    vectors: Dict[str, torch.Tensor] = {}
    vector_modes: Dict[str, str] = {}

    # =============================================================
    # 2) TRAIN-ONLY metrics
    # =============================================================
    if phase == "train":
        vectors = {
            "steps": steps.float()[halted_mask],
            "accuracy": per_seq_acc[halted_mask],
            "exact_accuracy": seq_correct[halted_mask].float(),
            "q_halt_accuracy_for_correct_output": q_halt_correct_output_vec,
            "q_halt_accuracy_for_wrong_output": q_halt_wrong_output_vec,
        }

        vector_modes = {
            "steps": "smooth",
            "accuracy": "smooth",
            "exact_accuracy": "smooth",
            "q_halt_accuracy_for_correct_output": "smooth",
            "q_halt_accuracy_for_wrong_output": "smooth",
        }

        scalar_modes.update({
            "accuracy": ("ignore",),
            "exact_accuracy": ("ignore",),
            "steps": ("ignore",),
        })

    # =============================================================
    # 3) EVAL-ONLY metrics
    # =============================================================
    elif phase == "eval":
        scalars.update({
            "accuracy": (per_seq_acc * valid_final).sum(),
            "exact_accuracy": (seq_correct & valid_final).sum(),
            "steps": (steps.float() * valid_final.float()).sum(),
        })

        scalar_modes.update({
            "accuracy": ("mean", "count"),
            "exact_accuracy": ("mean", "count"),
            "steps": ("mean", "count"),
        })

        vectors = {}
        vector_modes = {}

    # =============================================================
    return MetricRecord(
        scalars=scalars,
        scalar_modes=scalar_modes,
        vectors=vectors,
        vector_modes=vector_modes,
        phase=phase,
    )



