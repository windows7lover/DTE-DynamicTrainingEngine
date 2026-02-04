from __future__ import annotations

from dte.act.act_container import ACTContainer
from dte.interfaces import HaltingController
from dte.trainer.training_context import TrainingContext

def compute_loss(training_ctx: TrainingContext, container: ACTContainer):
    """
    Loss act_model_wrapper:
    Calls user-defined criterion on ACTContainer fields.
    """
    return training_ctx.criterion(
        container.outputs,
        container.model_memory,
        container.labels,
    )


def compute_metrics(training_ctx: TrainingContext, container: ACTContainer, loss_info: dict, phase: str):
    """
    Metrics act_model_wrapper:
    Calls user-defined compute_act_metrics() with correct halting info.
    """
    halting_state = container.halting_state
    halted_mask = training_ctx.halting_ctrl.get_halted_mask(halting_state)

    return training_ctx.train_state.metrics_fn(
        model_memory=container.model_memory,
        outputs=container.outputs,
        labels=container.labels,
        loss_info=loss_info,
        phase=phase,
        halted_mask=halted_mask,
        halting_state=halting_state,
    )


def update_halting(
    halting_ctrl: HaltingController,
    container: ACTContainer,
    loss_info: dict,
    training: bool,
):
    """
    Core-side halting update API.
    """

    new_state = halting_ctrl.update_halting_state(
        halting_state = container.halting_state,
        model_memory = container.model_memory,
        outputs = container.outputs,
        loss_info = loss_info,
        training = training,
    )

    container.halting_state = new_state
    return container