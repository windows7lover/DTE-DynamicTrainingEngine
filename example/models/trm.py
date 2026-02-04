"""
================================================================================
TINY RECURSIVE MODEL — CLEAN MODULAR IMPLEMENTATION
================================================================================
"""

from dataclasses import dataclass
from typing import List

import math
import torch
from torch import nn
from pydantic import BaseModel

from models.utils.common import trunc_normal_init_
from models.utils.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)


# ============================================================================
# INTERNAL MEMORY + OUTPUT STRUCTURES
# ============================================================================

@dataclass
class TRMMemory:
    prediction_y: torch.Tensor
    reasoning_Z: torch.Tensor
    steps: torch.Tensor

    def to(self, device: torch.device) -> "TRMMemory":
        return TRMMemory(
            prediction_y=self.prediction_y.to(device),
            reasoning_Z=self.reasoning_Z.to(device),
            steps=self.steps.to(device),
        )


@dataclass
class TRMOutputs:
    logits: torch.Tensor
    q_halt: torch.Tensor

    def reset(self, mask: torch.Tensor | None):
        if mask is None:
            self.logits = None
            self.q_halt = None
            return

        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return

        if self.logits is not None:
            self.logits[idx] = 0.0
        if self.q_halt is not None:
            self.q_halt[idx] = 0.0

    def to(self, device: torch.device) -> "TRMOutputs":
        return TRMOutputs(
            logits=self.logits.to(device) if self.logits is not None else None,
            q_halt=self.q_halt.to(device) if self.q_halt is not None else None,
        )


# ============================================================================
# TRM ARCHITECTURE CONFIG
# ============================================================================

class TRMConfig(BaseModel):
    seq_len: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False


# ============================================================================
# SUBCONFIGS (ENCODER / CORE / MEMORY INIT)
# ============================================================================

@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    forward_dtype: str


@dataclass
class CoreConfig:
    hidden_size: int
    num_heads: int
    seq_len: int
    expansion: float
    L_layers: int
    L_cycles: int
    H_cycles: int
    pos_encodings: str
    rope_theta: float
    forward_dtype: str
    mlp_t: bool
    rms_norm_eps: float


@dataclass
class MemoryInitConfig:
    seq_len: int
    hidden_size: int
    forward_dtype: str


# ============================================================================
# MODEL SUBMODULES
# ============================================================================

class TRMBlock(nn.Module):
   
    def __init__(
        self,
        hidden_size: int,
        seq_len: int,
        num_heads: int,
        expansion: float,
        mlp_t: bool,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.mlp_t_mode = mlp_t
        self.norm_eps = rms_norm_eps

        if mlp_t:
            self.mlp_t = SwiGLU(hidden_size=seq_len, expansion=expansion)
        else:
            self.self_attn = Attention(
                hidden_size=hidden_size,
                head_dim=hidden_size // num_heads,
                num_heads=num_heads,
                num_key_value_heads=num_heads,
                causal=False,
            )

        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.mlp_t_mode:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        out = self.mlp(hidden_states)
        return rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)


class TRMReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden: torch.Tensor, inj: torch.Tensor, **kwargs):
        hidden = hidden + inj
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, **kwargs)
        return hidden


class TRMEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.scale = math.sqrt(cfg.hidden_size)
        dt = getattr(torch, cfg.forward_dtype)

        self.embed = CastedEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            init_std=1.0 / self.scale,
            cast_to=dt,
        )

    def forward(self, input_ids: torch.Tensor):
        e = self.embed(input_ids.int())
        return self.scale * e


class TRMCore(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg

        blocks = [
            TRMBlock(
                hidden_size=cfg.hidden_size,
                seq_len=cfg.seq_len,
                num_heads=cfg.num_heads,
                expansion=cfg.expansion,
                mlp_t=cfg.mlp_t,
                rms_norm_eps=cfg.rms_norm_eps,
            )
            for _ in range(cfg.L_layers)
        ]

        self.core = TRMReasoningModule(blocks)

        # Heads (attached later by TRM)
        self.lm_head = None
        self.q_head = None

        if cfg.pos_encodings == "rope":
            self.rotary = RotaryEmbedding(
                dim=cfg.hidden_size // cfg.num_heads,
                max_position_embeddings=cfg.seq_len,
                base=cfg.rope_theta,
            )
        else:
            self.rotary = None

    def attach_heads(self, lm_head: nn.Module, q_head: nn.Module):
        self.lm_head = lm_head
        self.q_head = q_head

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def step(self, mem: TRMMemory, encoded: torch.Tensor):
        x = encoded
        cos_sin = self.rotary() if self.rotary is not None else None
        steps = mem.steps + 1

        pred = mem.prediction_y
        Z = mem.reasoning_Z

        # H cycles - 1 (no grad)
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                inj = pred + x
                for _ in range(self.cfg.L_cycles):
                    Z = self.core(Z, inj, cos_sin=cos_sin)
                pred = self.core(pred, Z, cos_sin=cos_sin)

        # final cycle WITH grad
        inj = pred + x
        for _ in range(self.cfg.L_cycles):
            Z = self.core(Z, inj, cos_sin=cos_sin)
        pred = self.core(pred, Z, cos_sin=cos_sin)

        new_mem = TRMMemory(
            prediction_y=pred.detach().clone(),
            reasoning_Z=Z.detach().clone(),
            steps=steps.detach().clone(),
        )

        logits = self.lm_head(pred)
        q_logits = self.q_head(pred[:, 0]).float()

        return new_mem, TRMOutputs(logits=logits, q_halt=q_logits[..., 0])


class TRMMemoryInitializer(nn.Module):
    def __init__(self, cfg: MemoryInitConfig):
        super().__init__()
        dt = getattr(torch, cfg.forward_dtype)
        D = cfg.hidden_size

        self.seq_len = cfg.seq_len
        self.hidden_size = cfg.hidden_size

        # UNLEARNABLE
        pred_init = trunc_normal_init_(torch.empty(1, 1, D, dtype=dt), std=1)
        Z_init    = trunc_normal_init_(torch.empty(1, 1, D, dtype=dt), std=1)
        self.register_buffer("pred_init", pred_init, persistent=True)
        self.register_buffer("Z_init", Z_init, persistent=True)

    def init_memory(self, inputs: torch.Tensor) -> TRMMemory:
        B = inputs.shape[0]
        L = self.seq_len
        D = self.hidden_size

        # detach is unnecessary now; clone keeps independent storage
        pred = self.pred_init.expand(B, L, D).clone()
        Z    = self.Z_init.expand(B, L, D).clone()
        steps = torch.zeros(B, dtype=torch.int32, device=inputs.device)

        return TRMMemory(prediction_y=pred, reasoning_Z=Z, steps=steps)

    def reset_memory(self, mem: TRMMemory, inputs: torch.Tensor, mask: torch.Tensor):
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return mem

        B = idx.numel()
        L = self.seq_len
        D = self.hidden_size

        pred = self.pred_init.expand(B, L, D).clone()
        Z    = self.Z_init.expand(B, L, D).clone()

        mem.prediction_y.index_copy_(0, idx, pred)
        mem.reasoning_Z.index_copy_(0, idx, Z)
        mem.steps.index_fill_(0, idx, 0)
        return mem



# ============================================================================
# MAIN TRM MODEL (ACT-compatible)
# ============================================================================

class TRM(nn.Module):
    def __init__(self, config_dict: dict, dataset_metadata):
        super().__init__()

        cfg = TRMConfig(**config_dict)
        cfg.seq_len = dataset_metadata.seq_len
        cfg.vocab_size = dataset_metadata.vocab_size

        self.config = cfg
        self.dataset_metadata = dataset_metadata

        # ---------- Split main config into sub-configs ----------
        enc_cfg, core_cfg, mem_cfg = self._split_config(cfg)

        # ---------- Build modules ----------
        self.encoder = TRMEncoder(enc_cfg)
        self.core = TRMCore(core_cfg)
        self.memory_init = TRMMemoryInitializer(mem_cfg)

        # Attach heads (core does not need vocab_size until now)
        lm_head = CastedLinear(cfg.hidden_size, cfg.vocab_size, bias=False)
        q_head = CastedLinear(cfg.hidden_size, 2, bias=True)
        self.core.attach_heads(lm_head, q_head)

    # ----------------------------------------------------------
    def _split_config(self, cfg: TRMConfig):
        enc = EncoderConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            forward_dtype=cfg.forward_dtype,
        )

        core = CoreConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            seq_len=cfg.seq_len,
            expansion=cfg.expansion,
            L_layers=cfg.L_layers,
            L_cycles=cfg.L_cycles,
            H_cycles=cfg.H_cycles,
            pos_encodings=cfg.pos_encodings,
            rope_theta=cfg.rope_theta,
            forward_dtype=cfg.forward_dtype,
            mlp_t=cfg.mlp_t,
            rms_norm_eps=cfg.rms_norm_eps,
        )

        mem = MemoryInitConfig(
            seq_len=cfg.seq_len,
            hidden_size=cfg.hidden_size,
            forward_dtype=cfg.forward_dtype,
        )

        return enc, core, mem
