"""Loss functions for classification, including hand-written versions to learn from.

A LOSS answers one question: "how wrong is the model on this batch?" with a single number.
Training = repeatedly nudging the weights to make that number smaller.

For classification the model outputs LOGITS (raw scores, one per class). The standard loss is
CROSS-ENTROPY: turn logits into probabilities with softmax, then take  -log(probability of the
correct class).  Confident and right -> loss ~ 0.  Confident and wrong -> loss is huge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_NAMES = ("cross_entropy", "label_smoothing", "focal")


# --------------------------------------------------------------------------- from scratch
def softmax(logits: torch.Tensor) -> torch.Tensor:
    """softmax(z)_i = exp(z_i) / sum_j exp(z_j), computed stably.

    Subtracting the max does not change the result (it cancels in the ratio) but prevents exp()
    from overflowing for large logits.
    """
    z = logits - logits.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=1, keepdim=True)


def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of  -log softmax(logits)[correct class].

    Uses log-sum-exp:  log softmax(z)_y = z_y - log(sum_j exp(z_j)),  which is more stable than
    computing softmax first and then taking log (that could hit log(0)).
    """
    z = logits - logits.max(dim=1, keepdim=True).values
    log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
    return -log_probs[torch.arange(len(targets)), targets].mean()


# --------------------------------------------------------------------------- modules
class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017):  FL = (1 - p_t)^gamma * CE.

    p_t is the probability the model gives the correct class. When the model is already confident
    and right (p_t near 1), the factor (1 - p_t)^gamma shrinks that example's loss toward 0, so
    training focuses on the HARD examples. gamma=0 gives plain cross-entropy. Popular for
    class-imbalanced problems (e.g. object detection with many easy background boxes).
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")  # -log p_t per example
        p_t = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


def build_loss(name: str, label_smoothing: float = 0.1, gamma: float = 2.0) -> nn.Module:
    """Return the loss module called `name` (see LOSS_NAMES).

    - cross_entropy   : the default, hard 0/1 targets.
    - label_smoothing : target = (1 - eps) * one-hot + eps / K (PyTorch's formulation, K = classes),
                        i.e. a little probability mass is spread over every class. This stops the model
                        from becoming over-confident and often generalises better.
    - focal           : down-weights easy examples (see FocalLoss).
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma)
    raise ValueError(f"unknown loss '{name}', choose one of {LOSS_NAMES}")
