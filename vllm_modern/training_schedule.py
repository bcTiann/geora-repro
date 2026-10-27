"""Single-learner batch arithmetic, independent of Torch and TRL imports."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingSchedule:
    micro_batch_size: int = 4
    generation_batch_size: int = 32
    answers_per_update: int = 128
    num_generations: int = 8

    def __post_init__(self):
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values.values()):
            raise ValueError("Batch counts must be positive integers")
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two completions per question")
        if (self.generation_batch_size % self.micro_batch_size
                or self.generation_batch_size % self.num_generations
                or self.answers_per_update % self.generation_batch_size):
            raise ValueError("Generation must be divisible by micro and G; update must be divisible by generation")

    @property
    def gradient_accumulation_steps(self):
        return self.answers_per_update // self.micro_batch_size

    @property
    def steps_per_generation(self):
        return self.generation_batch_size // self.micro_batch_size

    @property
    def rollouts_per_update(self):
        return self.answers_per_update // self.generation_batch_size

    def as_dict(self):
        return {**asdict(self), "learner_world_size": 1,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "steps_per_generation": self.steps_per_generation,
                "rollouts_per_update": self.rollouts_per_update,
                "questions_per_update": self.answers_per_update // self.num_generations}

    def versions(self, start_step, final_step):
        if not 0 <= start_step < final_step:
            raise ValueError("Require a nonempty optimizer update interval")
        publications = tuple(range(start_step, final_step + 1))
        generations = tuple(step for step in range(start_step, final_step)
                            for _ in range(self.rollouts_per_update))
        return publications, generations


def resolve_schedule(training, *, micro=None, generation=None, require_frozen=False):
    """Use the selected defaults; formal CLI overrides cannot drift silently."""
    result = TrainingSchedule(
        training["micro_batch_size"] if micro is None else micro,
        training["generation_batch_size"] if generation is None else generation,
        training["answers_per_update"], training["num_generations"])
    if require_frozen:
        if (result.micro_batch_size != training["micro_batch_size"]
                or result.generation_batch_size != training["generation_batch_size"]):
            raise ValueError("Formal batch settings must match the frozen configuration")
        for key in ("gradient_accumulation_steps", "steps_per_generation", "rollouts_per_update"):
            if key in training and training[key] != getattr(result, key):
                raise ValueError(f"Frozen derived schedule disagrees: {key}")
    return result
