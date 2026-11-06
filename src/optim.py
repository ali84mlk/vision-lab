"""Optimizers: the rule that turns gradients into weight updates.

The loss tells us HOW WRONG we are; backprop gives the gradient g = d(loss)/d(weights), i.e. the
direction in which the loss grows fastest. An optimizer decides how to step in the opposite direction.

    plain SGD :   w <- w - lr * g
    momentum  :   keep a running "velocity" so we roll through noisy/flat spots instead of zig-zagging
    Adam      :   momentum + a per-weight step size that adapts to how large that weight's gradients are
    AdamW     :   Adam with weight decay applied correctly ("decoupled"), the modern default

Hand-written versions of these live in `src/optim_scratch.py` and are explained in learn/02_optimizers.ipynb.
"""

from __future__ import annotations

from typing import Iterable

import torch

OPTIMIZER_NAMES = ("sgd", "sgd_momentum", "sgd_nesterov", "adam", "adamw")

# Sensible starting learning rates for fine-tuning a pretrained network.
# SGD-style optimizers take raw gradient steps and like a bigger LR; Adam-style rescale the gradient
# per weight, so their LR is smaller. The LR is the most important hyper-parameter: see learn/03.
DEFAULT_LR = {"sgd": 0.01, "sgd_momentum": 0.01, "sgd_nesterov": 0.01, "adam": 5e-4, "adamw": 5e-4}

# Typical weight decay per optimizer. Adam + L2 is awkward to tune (see learn/02), so it defaults to 0;
# AdamW's decoupled decay is meant to be used, and 0.01 is the usual starting value.
DEFAULT_WD = {"sgd": 5e-4, "sgd_momentum": 5e-4, "sgd_nesterov": 5e-4, "adam": 0.0, "adamw": 0.01}


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float | None = None,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Create the optimizer called `name` (see OPTIMIZER_NAMES). `lr=None` uses DEFAULT_LR[name].

    weight_decay discourages large weights (a form of regularisation):
      - SGD / Adam add it to the gradient (classic "L2 regularisation").
      - AdamW shrinks the weights directly, independent of the gradient scale. With Adam the L2 trick
        interacts badly with the adaptive step size, which is why AdamW exists. (See learn/02.)
    """
    if name not in OPTIMIZER_NAMES:
        raise ValueError(f"unknown optimizer '{name}', choose one of {OPTIMIZER_NAMES}")
    lr = DEFAULT_LR[name] if lr is None else lr

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "sgd_nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
