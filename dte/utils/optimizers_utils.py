import math

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
    ):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    lr_ratio = min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return base_lr * lr_ratio


def compute_lr(base_lr: float, scheduler_cfg, train_state):
    """
    scheduler_cfg is the Pydantic SchedulerConfig:
      - lr_warmup_steps
      - lr_min_ratio
      - (optionally num_cycles)
    """
    
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(scheduler_cfg["lr_warmup_steps"]),
        num_training_steps=train_state.total_training_steps,
        min_ratio=scheduler_cfg["lr_min_ratio"],
    )
