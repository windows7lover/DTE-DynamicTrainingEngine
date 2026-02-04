from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Stablemax helpers 
# ------------------------------------------------------------
def s(x, eps=1e-30):
    return torch.where(x < 0, 1 / (1 - x + eps), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, valid_mask):
    """
    Stablemax CE that masks invalid tokens.
    """
    logprobs = log_stablemax(logits.to(torch.float64))

    labels = labels.to(torch.int64)
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))

    gathered = torch.gather(logprobs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    return -torch.where(valid_mask, gathered, torch.zeros_like(gathered))


# ------------------------------------------------------------
# ACT Criterion (clean API: loss + flat info dict)
# ------------------------------------------------------------
class TRMCriterion(nn.Module):
    """
    Unified loss + metrics + halting-signals.

    forward() returns:
      - loss : scalar Tensor
      - info : flat dict containing:
            loss/*
            halt/*
            metric/*
    """

    def __init__(self, dataset_metadata: DatasetMetadata, loss_type: str):
        super().__init__()
        self.dataset_metadata = dataset_metadata

        # normalize alias names
        lt = loss_type.lower()
        self.loss_type = {
            "stablemax": "stablemax",
            "stablemax_cross_entropy": "stablemax",
            "softmax": "softmax",
            "softmax_cross_entropy": "softmax",
        }.get(lt, lt)

    # ------------------------------------------------------------
    # Main criterion
    # ------------------------------------------------------------
    def forward(self, output, model_memory, labels):
        """
        Args:
            output: TRMOutputs(logits, q_halt, ...)
            model_memory: TRMInnerCarry (unused except to show consistency / future use)
            labels: Tensor[B, L]

        Returns:
            loss: scalar
            info: dict[str, Tensor]
        """

        logits = output.logits          # [B, L, V]
        q_halt_logits = output.q_halt   # [B] or [B,1]

        pad_id = self.dataset_metadata.pad_id
        valid_mask = labels != pad_id

        B, L, V = logits.shape

        # reshape for CE
        flat_logits = logits.reshape(B * L, V)
        flat_labels = labels.reshape(B * L)
        flat_valid = valid_mask.reshape(B * L)

        # LM loss
        if self.loss_type == "stablemax":
            flat_ce = stablemax_cross_entropy(flat_logits, flat_labels, flat_valid)
        else:
            flat_ce = F.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=pad_id,
                reduction="none",
            )

        per_token_ce = flat_ce.reshape(B, L)
        token_count = valid_mask.sum(-1).clamp(min=1)
        lm_loss = (per_token_ce.sum(-1) / token_count).sum()

        # correctness signals
        preds = torch.argmax(logits, dim=-1)
        token_correct = (preds == labels) & valid_mask
        seq_correct = (token_correct.sum(-1) == token_count)

        # halting loss
        q_h = q_halt_logits.squeeze(-1)
        halt_loss = F.binary_cross_entropy_with_logits(
            q_h, seq_correct.float(), reduction="sum"
        )

        loss = lm_loss + 0.5 * halt_loss

        # metrics
        token_acc = (
            token_correct.float().sum() / valid_mask.float().sum()
            if valid_mask.any()
            else torch.tensor(0.0)
        )
        seq_acc = seq_correct.float().mean()

        info = {
            "loss/lm": lm_loss.detach(),
            "loss/halt": halt_loss.detach(),
            "loss/total": loss.detach(),
            "halt/seq_correct": seq_correct,
            "halt/token_correct": token_correct,
            "metric/seq_acc": seq_acc.detach(),
            "metric/token_acc": token_acc.detach(),
        }

        return loss, info

