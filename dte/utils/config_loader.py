# config_loader.py
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
import os
from pydantic import BaseModel
import importlib


# ================================================================
# DYNAMIC IMPORT
# ================================================================
def import_from_path(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


# ================================================================
# WANDB CONFIGS
# ================================================================
class WandbConfig(BaseModel):
    enabled: bool = True
    mode: str = "online"
    entity: str | None = None
    project_name: str | None = None
    run_name: str | None = None
    resume: str = "never"
    tags: List[str] = []
    log_code: bool = True
    log_git: bool = True


# ================================================================
# DATA CONFIGS
# ================================================================
class OnlineDatasetConfig(BaseModel):
    generator_cls_path: str
    generator: Dict[str, Any]
    train_size: int | None = None
    eval_size: int | None = None


class FixedDatasetConfig(BaseModel):
    train_path: str | None = None
    eval_path: str | None = None


class DatasetConfig(BaseModel):
    mode: str

    dataset_metadata_cls_path: str
    dataset_metadata: Dict[str, Any]

    online: OnlineDatasetConfig | None = None
    fixed: FixedDatasetConfig | None = None


# ================================================================
# ARCH / MODEL CONFIG
# ================================================================
class ArchConfig(BaseModel):
    name: str

    # Architecture class
    model_cls_path: str
    model_cls: Any = None

    # ACT runtime components (must exist!)
    output_cls_path: str
    halting_controller_cls_path: str
    criterion_cls_path: str
    metrics_fn_path: str

    output_cls: Any = None
    halting_controller_cls: Any = None
    criterion_cls: Any = None
    metrics_fn: Any = None

    # Optional kwargs file
    config_file: str | None = None
    component_config: dict = {}
    
class CompileOptions(BaseModel):
    enabled: bool | None = None
    backend: str | None = None
    mode: str | None = None
    fullgraph: bool | None = None
    dynamic: bool | None = None
    options: Dict[str, Any] | None = None

# ================================================================
# COMPILE CONFIG (NEW INTERFACE)
# ================================================================

class TorchCompileConfig(BaseModel):
    enabled: bool = True
    
    default: CompileOptions = CompileOptions(
        enabled=True       # per-function default ON
    )

    # default options
    default: CompileOptions = CompileOptions()

    # per-function overrides
    encode: CompileOptions = CompileOptions()
    step_encoded: CompileOptions = CompileOptions()
    step_once: CompileOptions = CompileOptions()
    unroll_train: CompileOptions = CompileOptions()
    unroll_eval: CompileOptions = CompileOptions()
    

# ================================================================
# LOAD/SAVE CONFIGS
# ================================================================
class CheckpointSaveConfig(BaseModel):
    train: bool = True
    model: bool = False
    ema: bool = False


class CheckpointLoadConfig(BaseModel):
    path: str | None = None


class CheckpointOverrideConfig(BaseModel):
    model_path: str | None = None
    ema_path: str | None = None
    reset_optimizer: bool = True
    reset_step: bool = True


class CheckpointConfig(BaseModel):
    path: str
    auto_resume: bool = True
    save: CheckpointSaveConfig = CheckpointSaveConfig()
    load: CheckpointLoadConfig = CheckpointLoadConfig()
    override: CheckpointOverrideConfig = CheckpointOverrideConfig()


# ================================================================
# TRAINING SUBCONFIGS
# ================================================================
class OptimizerConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class SchedulerConfig(BaseModel):
    name: str
    lr_min_ratio: float
    lr_warmup_steps: int
    lr: float


class EMAConfig(BaseModel):
    use_ema: bool
    ema_rate: float | None = None


class ACTConfig(BaseModel):
    unroll_steps_train: int
    unroll_steps_eval: int
    learn_encoder: bool
    learn_init_memory: bool


# ================================================================
# MASTER TRAINING CONFIG
# ================================================================
class PretrainConfig(BaseModel):

    arch: ArchConfig
    dataset: DatasetConfig

    training: Dict[str, Any]
    
    torch_compile: TorchCompileConfig | None = None

    batch_size: int
    epochs: int
    max_num_step_per_train_epoch: int | None = None
    max_num_step_per_eval_epoch: int | None = None

    wandb_config: WandbConfig | None = None

    seed: int = 0
    eval_interval: Optional[int] = None
    eval_start_epoch: Optional[int] = 0
    
    checkpoint: CheckpointConfig | None = None

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "PretrainConfig":
        raw = OmegaConf.to_container(cfg, resolve=True)
        config = cls(**raw)

        # ------------------------------------------------------------
        # 1. Import ONLY the model class
        # ------------------------------------------------------------
        arch = config.arch
        arch.model_cls = import_from_path(arch.model_cls_path)
        arch.output_cls = import_from_path(arch.output_cls_path)
        arch.halting_controller_cls = import_from_path(arch.halting_controller_cls_path)
        arch.criterion_cls = import_from_path(arch.criterion_cls_path)
        arch.metrics_fn = import_from_path(arch.metrics_fn_path)


        # ------------------------------------------------------------
        # 2. Load optional component config YAML for model kwargs
        # ------------------------------------------------------------
        if arch.config_file:
            comp_cfg = OmegaConf.load(arch.config_file)
            arch.component_config = OmegaConf.to_container(comp_cfg, resolve=True)
        else:
            arch.component_config = {}

        # ------------------------------------------------------------
        # 3. wandb metadata injection
        # ------------------------------------------------------------
        if config.wandb_config:
            config.wandb_config = WandbConfig(**raw["wandb_config"])

        # ------------------------------------------------------------
        # 4. auto checkpoint directory
        # ------------------------------------------------------------
        if config.checkpoint is None:
            config.checkpoint = CheckpointConfig(
                path=os.path.join(
                    "checkpoints",
                    config.project_name or "project",
                    config.run_name or "run",
                )
            )

        return config

    @classmethod
    def sync(cls, config, rank, world_size):
        objs = [config if rank == 0 else None]
        if world_size > 1:
            dist.broadcast_object_list(objs, src=0)
        return objs[0]
