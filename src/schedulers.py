"""Learning-rate schedules: changing the LR during training.

A big LR makes fast progress early but bounces around the minimum later; a small LR is precise but slow.
Schedules give you both: start bigger, end smaller (and often start with a short WARMUP, because at
step 0 the weights/Adam statistics are not yet reliable and a large LR can wreck a pretrained model).

All schedulers here are stepped ONCE PER BATCH (not per epoch) so the curve is smooth.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR, StepLR

SCHEDULER_NAMES = ("constant", "step", "cosine", "warmup_cosine", "onecycle")


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Multiplier on the base LR at `step`: linear 0 -> 1 during warmup, then cosine 1 -> 0.

        warmup:  factor = step / warmup_steps
        decay :  factor = 0.5 * (1 + cos(pi * progress)),  progress in [0, 1] after warmup
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 1.0,
    gamma: float = 0.1,
):
    """Create the schedule called `name` (see SCHEDULER_NAMES). Call `scheduler.step()` after every batch.

    - constant      : LR never changes (the baseline to compare against).
    - step          : multiply LR by `gamma` every ~third of training (classic "staircase").
    - cosine        : smooth decay from base LR to ~0 following half a cosine.
    - warmup_cosine : linear warmup for `warmup_epochs`, then cosine decay. A strong default.
    - onecycle      : LR ramps up to the base LR then anneals far below it (super-convergence).
    """
    if name not in SCHEDULER_NAMES:
        raise ValueError(f"unknown scheduler '{name}', choose one of {SCHEDULER_NAMES}")
    total = max(1, epochs * steps_per_epoch)

    if name == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)
    if name == "step":
        return StepLR(optimizer, step_size=max(1, (epochs // 3) * steps_per_epoch), gamma=gamma)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total)
    if name == "warmup_cosine":
        warmup = int(warmup_epochs * steps_per_epoch)
        return LambdaLR(optimizer, lambda step: warmup_cosine_factor(step, warmup, total))

    # onecycle: it also cycles momentum / beta1 by default. Only enable that when the optimizer really
    # uses momentum, otherwise it would silently turn plain SGD into momentum SGD.
    defaults = optimizer.defaults
    uses_momentum = "betas" in defaults or defaults.get("momentum", 0) > 0
    max_lr = [g["lr"] for g in optimizer.param_groups]
    return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total, pct_start=0.3, cycle_momentum=uses_momentum)


def simulate_schedule(name: str, base_lr: float, epochs: int, steps_per_epoch: int, **kwargs) -> list[float]:
    """The LR used at every step, computed on a dummy parameter. Handy for plotting (learn/03)."""
    dummy = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([dummy], lr=base_lr, momentum=0.9)
    sched = build_scheduler(name, opt, epochs, steps_per_epoch, **kwargs)
    lrs = []
    for _ in range(epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs
