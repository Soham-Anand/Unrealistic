import math
import mlx.core as mx


class CosineScheduler:
    def __init__(self, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float = 1e-6):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)

        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.peak_lr - self.min_lr)
