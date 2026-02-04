import os
import yaml
import subprocess
import logging
from typing import Optional

import wandb
from torch import nn

from dte.utils.config_loader import PretrainConfig, WandbConfig


log = logging.getLogger(__name__)


# ================================================================
# Git utilities
# ================================================================
def get_git_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None
        
def _config_for_serialization(cfg: PretrainConfig) -> dict:
    data = cfg.model_dump()

    # Remove runtime-injected objects
    arch = data.get("arch", {})
    arch.pop("model_cls", None)
    arch.pop("output_cls", None)
    arch.pop("halting_controller_cls", None)
    arch.pop("criterion_cls", None)
    arch.pop("metrics_fn", None)

    return data



# ================================================================
# W&B initialization
# ================================================================
def init_wandb(
    wandb_cfg: WandbConfig,
    full_config: PretrainConfig,
    model: nn.Module,
) -> None:
    """
    Initialize Weights & Biases.

    Explicit behavior:
    - WandB is logging/metadata ONLY
    - All checkpointing is handled internally
    - Called exactly once, on master rank
    """

    # ------------------------------------------------------------
    # 1. Hard guard
    # ------------------------------------------------------------
    if not wandb_cfg.enabled:
        log.warning(
            "W&B is DISABLED (wandb_config.enabled = False). "
            "No metrics, configs, or code will be logged."
        )
        return

    # ------------------------------------------------------------
    # 2. Mode handling
    # ------------------------------------------------------------
    os.environ["WANDB_MODE"] = wandb_cfg.mode

    if wandb_cfg.mode == "offline":
        log.warning(
            "W&B running in OFFLINE mode. "
            "Runs will be logged locally and must be synced manually."
        )
    elif wandb_cfg.mode == "disabled":
        log.warning(
            "wandb_config.mode='disabled'. "
            "This overrides enabled=True. W&B will not initialize."
        )
        return

    # ------------------------------------------------------------
    # 3. Checkpoint responsibility warning
    # ------------------------------------------------------------
    log.info(
        "W&B checkpointing is DISABLED. "
        "All training state (model / optimizer / EMA / step) "
        "is handled by the internal checkpoint system."
    )

    # ------------------------------------------------------------
    # 4. Init
    # ------------------------------------------------------------
    wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project_name,
        name=wandb_cfg.run_name,
        resume=wandb_cfg.resume,
        tags=wandb_cfg.tags,
        config=full_config.model_dump(),
    )

    log.info(
        "Initialized W&B run: project='%s', name='%s'",
        wandb_cfg.project_name,
        wandb_cfg.run_name,
    )

    # ------------------------------------------------------------
    # 5. Static run metadata
    # ------------------------------------------------------------
    num_params = sum(p.numel() for p in model.parameters())
    wandb.log({"num_params": num_params}, step=0)

    # ------------------------------------------------------------
    # 6. Git hash
    # ------------------------------------------------------------
    if wandb_cfg.log_git:
        git_hash = get_git_hash()
        if git_hash is not None:
            wandb.config.update(
                {"git_hash": git_hash},
                allow_val_change=True,
            )
        else:
            log.warning("Could not resolve git hash; repository may be dirty or missing.")

    # ------------------------------------------------------------
    # 7. Config snapshot
    # ------------------------------------------------------------
    if full_config.checkpoint is None:
        log.warning(
            "No checkpoint config found. "
            "Full config will NOT be snapshotted to disk."
        )
        return

    if wandb.run is None:
        log.warning(
            "W&B run is not initialized; skipping config snapshot and code logging."
        )
        return

    ckpt_path = full_config.checkpoint.path
    os.makedirs(ckpt_path, exist_ok=True)

    config_path = os.path.join(ckpt_path, "full_config.yaml")
    with open(config_path, "wt") as f:
        yaml.safe_dump(_config_for_serialization(full_config), f)

    log.info("Saved full config snapshot to %s", config_path)

    # ------------------------------------------------------------
    # 8. Code logging
    # ------------------------------------------------------------
    if wandb_cfg.log_code:
        log.info(
            "Logging code snapshot to W&B (READ-ONLY). "
            "This does NOT affect checkpointing or resume behavior."
        )
        wandb.run.log_code(ckpt_path)
