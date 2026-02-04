from dataclasses import dataclass

# TODO: separate this from train state

@dataclass(frozen=True)
class TrainingSchedule:
    max_epoch: int
    max_training_steps_per_epoch: int
    max_eval_steps_per_epoch: int

    @property
    def total_training_steps(self) -> int:
        return self.max_epoch * self.max_training_steps_per_epoch
