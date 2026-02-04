from __future__ import annotations

import os

import hydra
import torch
import torch.distributed as dist
import tqdm
import wandb

from dte.trainer.api import compute_loss, compute_metrics, update_halting
from dte.trainer.trainer_builder import TrainerBuilder
from dte.utils.distributed import master_print, teardown_distributed, ddp_check_grad_synced
from dte.utils.model_summary import print_param_table

# =============================================================================
# HELPERS
# =============================================================================
def should_evaluate_epoch(cfg, epoch: int) -> bool:
    return (
        cfg.eval_interval
        and epoch >= cfg.eval_start_epoch
        and (epoch + 1) % cfg.eval_interval == 0
    )


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


# =============================================================================
# UNIFIED ACT STEP (TRAIN + EVAL)
# =============================================================================
def act_step(training_ctx, phase: str):
    train_state = training_ctx.train_state
    cfg = train_state.config
    dist_ctx = training_ctx.dist
    data = train_state.data_module

    is_train = (phase == "train")
    mode = "train" if is_train else "eval"

    # Select stream explicitly
    stream = data.train_stream if is_train else data.eval_stream
    assert stream is not None, "DataModule.setup() must be called before act_step()."
    
    container = training_ctx.act_manager.ensure_container_and_refresh_halted(container=train_state.container, stream=stream)
    container, num_act_steps = training_ctx.act_manager.run(container, mode=mode)
    train_state.container = container

    # Loss
    loss, loss_info = compute_loss(training_ctx, container)

    # Backward + optimizer
    if is_train:
        loss = loss
        loss.backward()
        # ddp_check_grad_synced(train_state.model, dist_ctx)
        lr = training_ctx.optimizer.step(train_state)
    else:
        lr = None

    # Halting update
    container = update_halting(
        training_ctx.halting_ctrl,
        container,
        loss_info,
        training=is_train,
    )
    train_state.container = container

    # Metrics
    rec = compute_metrics(training_ctx, container, loss_info, phase=phase)
    metrics = training_ctx.metric_helper.process_metrics(
        rec,
        rank=dist_ctx.rank,
        extra_scalars={"train/lr": lr} if is_train else None,
    )

    return metrics, num_act_steps



# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_epoch(training_ctx, epoch: int):
    train_state = training_ctx.train_state
    dist_ctx = training_ctx.dist
    data_module = train_state.data_module
    cfg = train_state.config

    train_state.epoch = epoch
    train_state.model.train()
    train_state.container = None  # reset rolling container each epoch

    data_module.on_epoch_start(epoch)

    target_steps = train_state.max_training_steps_per_epoch
    acc_steps = 0

    bar = tqdm.tqdm(total=target_steps, desc=f"Epoch {epoch}") if dist_ctx.is_master else None

    while acc_steps < target_steps:
        metrics, num_act_steps = act_step(training_ctx, phase="train")

        acc_steps += num_act_steps
        train_state.step += num_act_steps

        if dist_ctx.is_master and metrics:
            if cfg.wandb_config.enabled:
                wandb.log(metrics, step=train_state.step)
            if bar:
                bar.update(min(num_act_steps, target_steps - bar.n))

        # EMA: always update using unwrapped model for stable parameter names
        if training_ctx.ema:
            training_ctx.ema.update(_unwrap(train_state.model))

    if bar:
        bar.close()


# =============================================================================
# EVALUATION LOOP
# =============================================================================
def evaluate(training_ctx):
    train_state = training_ctx.train_state
    model = train_state.model
    dist_ctx = training_ctx.dist
    cfg = train_state.config
    train_state.container = None  # reset rolling container

    max_steps = train_state.max_eval_steps_per_epoch
    master_print(dist_ctx, "EVALUATE")

    acc = []
    bar = tqdm.tqdm(total=max_steps, desc="Eval") if dist_ctx.is_master else None

    def run_eval_loop(eval_model):
        eval_model.eval()
        acc_steps = 0

        while acc_steps < max_steps:
            metrics, num_act_steps = act_step(training_ctx, phase="eval")
            acc.append(metrics)

            acc_steps += num_act_steps
            if bar:
                bar.update(min(num_act_steps, max_steps - bar.n))

    with torch.inference_mode():
        if training_ctx.ema is not None:
            # swap() unwraps internally, but we pass unwrapped anyway for consistency
            m = _unwrap(model)
            with training_ctx.ema.swap(m, strict=False):
                run_eval_loop(m)
        else:
            run_eval_loop(_unwrap(model))

    if bar:
        bar.close()

    final = {}
    count = 0
    for m in acc:
        if m is None:
            continue
        count += 1
        for k, v in m.items():
            final[k] = final.get(k, 0.0) + v

    if count > 0:
        for k in final:
            final[k] /= count

    return final


# =============================================================================
# TRAIN LOOP
# =============================================================================
def run_training(training_ctx):
    train_state = training_ctx.train_state
    cfg = train_state.config
    dist_ctx = training_ctx.dist

    start_epoch = train_state.epoch

    if dist_ctx.is_master:
        print(
            f"[train] starting epoch loop at epoch={start_epoch} "
            f"(ckpt epoch={train_state.epoch}, step={train_state.step})",
            flush=True,
        )

    for epoch in range(start_epoch, cfg.epochs):
        train_epoch(training_ctx, epoch)
        train_state.epoch = epoch + 1

        if should_evaluate_epoch(cfg, epoch):
            metrics = evaluate(training_ctx)
            if dist_ctx.is_master and metrics:
                if cfg.wandb_config.enabled:
                    wandb.log(metrics, step=train_state.step)
                if cfg.checkpoint.save.train:
                    training_ctx.checkpoint_manager.save()

    master_print(dist_ctx, "Training complete.")


# =============================================================================
# HYDRA LAUNCH
# =============================================================================
@hydra.main(config_path="config", config_name="config", version_base=None)
def launch(hydra_cfg):
    builder = TrainerBuilder.from_hydra(hydra_cfg)

    training_ctx = None
    try:
        if builder.dist.is_master:
            print("Starting TRM pretraining", flush=True)

        training_ctx = builder.build()
        dist_ctx = training_ctx.dist

        if dist_ctx.is_master:
            print_param_table(training_ctx.train_state.model)

        # One sync point before training starts (safe backend)
        dist_ctx.barrier(backend="gloo")

        run_training(training_ctx)

    finally:
        teardown_distributed(training_ctx.dist if training_ctx is not None else builder.dist)



if __name__ == "__main__":
    launch()

