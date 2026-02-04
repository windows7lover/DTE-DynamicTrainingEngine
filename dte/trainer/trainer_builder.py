from __future__ import annotations

import os
import torch

from dte.utils.distributed import setup_distributed, DistContext, master_print
from dte.utils.wandb_utils import init_wandb
from dte.trainer.train_state import TrainingState
from dte.trainer.optimization_helper import OptimizationHelper
from dte.trainer.metric_helper import MetricHelper
from dte.trainer.training_context import TrainingContext
from dte.trainer.checkpoint import CheckpointManager
from dte.trainer.ema import EMAHelper
from dte.act.act_model_factory import ACTModelFactory, ACTModelCompiler
from dte.act.act_container import ACTContainerManager
from dte.data.data_module import DataModule
from dte.utils.config_loader import PretrainConfig


class TrainerBuilder:
    """
    Centralizes the entire initialization pipeline.
    """

    def __init__(self, config: PretrainConfig, hydra_cfg):
        self.config = config
        self.hydra_cfg = hydra_cfg

        # distributed
        self.dist = setup_distributed()

        # rank-local device (use LOCAL_RANK from DistContext)
        self.device = self.dist.device
        torch.random.manual_seed(config.seed + self.dist.rank)

    # ------------------------------------------------------------------
    @classmethod
    def from_hydra(cls, hydra_cfg):
        cfg = PretrainConfig.sync(
            PretrainConfig.from_hydra(hydra_cfg),
            hydra_cfg.rank if "rank" in hydra_cfg else 0,
            hydra_cfg.world_size if "world_size" in hydra_cfg else 1,
        )
        return cls(cfg, hydra_cfg)

    # ------------------------------------------------------------------
    def build_data(self):
        data_module = DataModule(
            dataset_config=self.config.dataset,
            batch_size=self.config.batch_size,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            seed=self.config.seed,
            device=self.device,
        )
        data_module.setup()
        return data_module

    # ------------------------------------------------------------------
    def build_train_state(self, data_module):

        config = self.config
        # Resolve per-epoch limits and check for None
        # TRAIN
        max_train = resolve_epoch_steps(
            size=data_module.train_size,
            limit=config.max_num_step_per_train_epoch,
            name="train",
        )

        # EVAL
        max_eval = resolve_epoch_steps(
            size=data_module.eval_size,
            limit=config.max_num_step_per_eval_epoch,
            name="eval",
        )

        train_state = TrainingState(
            config=config,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            max_epoch=config.epochs,
            max_training_steps_per_epoch=max_train,
            max_eval_steps_per_epoch=max_eval,
            data_module=data_module,
            device=self.device,
        )

        return train_state

    # ------------------------------------------------------------------
    def build_model(self, train_state: TrainingState, data_module):
        arch = self.config.arch
        cfg = arch.component_config.get("model_config", {}).copy()

        cfg["seq_len"] = data_module.dataset_metadata.seq_len
        cfg["vocab_size"] = data_module.dataset_metadata.vocab_size

        device = train_state.device

        model = arch.model_cls(
            config_dict=cfg,
            dataset_metadata=data_module.dataset_metadata,
        ).to(device)

        train_state.model = model

    # ------------------------------------------------------------------
    def build_optimizer(self, train_state: TrainingState):
        cfg = self.config.training["optimizer"]
        sched = self.config.training["lr_scheduler"]
        opt = OptimizationHelper(
            optimizer_cfg=cfg,
            scheduler_cfg=sched,
            grad_clip=self.config.training.get("grad_clip", None),
        )
        return opt

    # ------------------------------------------------------------------
    def build_ema(self, train_state: TrainingState):
        if not self.config.training["ema"]["use_ema"]:
            return None
        ema = EMAHelper(mu=self.config.training["ema"]["ema_rate"])
        ema.register(train_state.model)
        return ema

    # ------------------------------------------------------------------
    def build_act_manager(self, train_state: TrainingState):
        """
        Use ACTModelFactory to assemble:
          base_model -> ACTModelWrapper -> optional DDP(ACTModelWrapper)
        and store the unwrapped ACT wrapper explicitly for post-resume compile.
        """
        arch = self.config.arch
        device = train_state.device

        global_bs = self.config.batch_size
        world_size = self.dist.world_size
        assert global_bs % world_size == 0, (
            f"config.batch_size={global_bs} must be divisible by world_size={world_size}"
        )
        per_rank_bs = global_bs // world_size

        base_model = train_state.model
        assert base_model is not None

        assembled = ACTModelFactory.build(
            base_model=base_model,
            device=device,
            dist=self.dist,
            train_unroll_steps=self.config.training["act"]["unroll_steps_train"],
            eval_unroll_steps=self.config.training["act"]["unroll_steps_eval"],
            ddp_find_unused_parameters=False,
            ddp_static_graph=False,  # keep False unless proven safe
        )

        # From now on, the train model is the wrapper (possibly DDP)
        train_state.model = assembled.train_model

        # Persist unwrapped wrapper for init/reset + compile (DDP doesn't proxy attrs)
        train_state.act_wrapper = assembled.inner_wrapper  # type: ignore[attr-defined]

        act_manager = ACTContainerManager(
            halting_ctrl=arch.halting_controller_cls(
                **arch.component_config.get("halting_config", {})
            ),
            init_memory_fn=assembled.inner_wrapper.init_memory,
            reset_memory_fn=assembled.inner_wrapper.reset_memory,
            model_wrapper=assembled.train_model,  # keep DDP wrapper here; run() calls mw(mem,x)
            device=device,
            batch_size=per_rank_bs,
        )
        return act_manager

    # ------------------------------------------------------------------
    def build_criterion(self, data_module):
        arch = self.config.arch
        criterion_cls = arch.criterion_cls
        return criterion_cls(
            dataset_metadata=data_module.dataset_metadata,
            **arch.component_config.get("criterion_config", {}),
        )

    # ------------------------------------------------------------------
    def build_metrics_fn(self):
        arch = self.config.arch
        return arch.metrics_fn

    # ------------------------------------------------------------------
    def build_metric_helper(self):
        return MetricHelper(
            dist_ctx=self.dist,
            window_size=self.config.batch_size,
        )

    # ------------------------------------------------------------------
    def build_training_ctx(self, train_state, opt, ema, act_manager, metric_helper, criterion):
        halting_ctrl = act_manager.halting_ctrl

        return TrainingContext(
            train_state=train_state,
            dist=self.dist,
            optimizer=opt,
            metric_helper=metric_helper,
            act_manager=act_manager,
            criterion=criterion,
            halting_ctrl=halting_ctrl,
            ema=ema,
        )

    # ------------------------------------------------------------------
    def build(self):
        master_print(self.dist, "=== Creating data module ===")
        data = self.build_data()

        master_print(self.dist, "=== Initializing TrainingState ===")
        train_state = self.build_train_state(data)

        master_print(self.dist, "=== Building model ===")
        self.build_model(train_state, data)
        act_manager = self.build_act_manager(train_state)

        master_print(self.dist, "=== Building optimizer / helpers ===")
        opt = self.build_optimizer(train_state)
        ema = self.build_ema(train_state)
        metric_helper = self.build_metric_helper()
        criterion = self.build_criterion(data)
        train_state.metrics_fn = self.build_metrics_fn()

        master_print(self.dist, "=== Creating TrainingContext ===")
        training_ctx = self.build_training_ctx(
            train_state, opt, ema, act_manager, metric_helper, criterion
        )

        # ----------------------------------------------------
        # Checkpoint manager
        # ----------------------------------------------------
        ckpt_cfg = self.config.checkpoint
        if ckpt_cfg is None:
            raise RuntimeError("config.checkpoint must not be None")

        training_ctx.checkpoint_manager = CheckpointManager(
            ctx=training_ctx,
            base_path=ckpt_cfg.path,
        )

        # ----------------------------------------------------
        # Resume logic
        # ----------------------------------------------------
        if ckpt_cfg.load.path is not None:
            training_ctx.checkpoint_manager.load(ckpt_cfg.load.path)
        elif ckpt_cfg.auto_resume:
            last = os.path.join(ckpt_cfg.path, "train", "last.pt")
            if os.path.exists(last):
                training_ctx.checkpoint_manager.load(last)

        # ----------------------------------------------------
        # Post-load overrides
        # ----------------------------------------------------
        ov = ckpt_cfg.override
        ts = training_ctx.train_state
        model = ts.model.module if hasattr(ts.model, "module") else ts.model

        # ---- override model
        if ov.model_path is not None:
            model_ckpt = torch.load(ov.model_path, map_location="cpu")
            model.load_state_dict(model_ckpt["model"], strict=True)

            # reset optimizer if requested
            if ov.reset_optimizer:
                print(
                    "[checkpoint override] reset_optimizer=True -> optimizer state cleared"
                )
                training_ctx.optimizer.optimizer = None

            # reset step / epoch if requested
            if ov.reset_step:
                print(
                    "[checkpoint override] reset_step=True -> step and epoch reset to 0"
                )
                ts.step = 0
                ts.epoch = 0

            # EMA must be re-registered from model
            if training_ctx.ema:
                training_ctx.ema.register(ts.model)

        # ---- override EMA only
        if ov.ema_path is not None and training_ctx.ema:
            ema_ckpt = torch.load(ov.ema_path, map_location="cpu")
            training_ctx.ema.load_state_dict(ema_ckpt["ema"])

        # ----------------------------------------------------
        # Initialize W&B
        # ----------------------------------------------------
        if (
            self.dist.is_master
            and self.config.wandb_config
            and self.config.wandb_config.enabled
        ):
            init_wandb(self.config.wandb_config, self.config, train_state.model)

        # ----------------------------------------------------
        # Compile ACT Model (post-resume)
        # ----------------------------------------------------
        if self.config.torch_compile and self.config.torch_compile.enabled:
            master_print(self.dist, "=== Compiling ACT functions (post-resume) ===")

            # compile the unwrapped ACTModelWrapper (DDP won't expose encode/step_*)
            act_wrapper = getattr(train_state, "act_wrapper", None)
            if act_wrapper is None:
                raise RuntimeError(
                    "train_state.act_wrapper missing; build_act_manager() must set it."
                )

            ACTModelCompiler.compile(
                wrapper=act_wrapper,
                torch_compile_cfg=self.config.torch_compile,
                dist=self.dist,
            )

        return training_ctx


def resolve_epoch_steps(size: int | None, limit: int | None, name: str) -> int:
    """
    Resolve the number of steps for one epoch.

    Rules:
        - If both size and limit are None -> error.
        - If limit is given, must be >= 1.
        - If size is None  -> require limit; return limit.
        - If limit is None -> return size.
        - If both provided -> return min(size, limit).
    """

    if size is None and limit is None:
        raise ValueError(
            f"Cannot determine number of {name} steps per epoch: "
            f"both dataset size and max_num_step_per_{name}_epoch are None."
        )

    if limit is not None and limit < 1:
        raise ValueError(
            f"max_num_step_per_{name}_epoch must be >= 1 (got {limit})."
        )

    if size is None:
        # Infinite dataset → limit must exist
        return limit  # type: ignore[return-value]

    if limit is None:
        # Finite dataset, no limit
        return size

    # Both exist → capped epoch
    return min(size, limit)
