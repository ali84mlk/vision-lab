"""SGD (+momentum, Nesterov), Adam and AdamW written by hand, ~10 lines each.

They mirror the torch.optim API (`step()`, `zero_grad()`) and use the SAME formulas as PyTorch, so
learn/02_optimizers.ipynb can prove they produce identical updates. Reading these next to
`torch.optim.SGD/Adam/AdamW` is the fastest way to demystify them.
"""

from __future__ import annotations

from typing import Iterable

import torch


class _Base:
    def __init__(self, params: Iterable[torch.Tensor]):
        self.params = [p for p in params if p.requires_grad]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class SGDScratch(_Base):
    """Stochastic gradient descent, optionally with (Nesterov) momentum and L2 weight decay.

        g   = grad + weight_decay * w                  (L2 regularisation)
        v   = momentum * v + g                         (velocity: decaying sum of past gradients)
        w  -= lr * v                                   (plain momentum)
        w  -= lr * (g + momentum * v)                  (Nesterov: "look ahead" along the velocity)

    With momentum=0 this is plain SGD:  w -= lr * g.
    """

    def __init__(self, params, lr: float, momentum: float = 0.0, nesterov: bool = False, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr, self.momentum, self.nesterov, self.weight_decay = lr, momentum, nesterov, weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p if self.weight_decay else p.grad
            if self.momentum:
                v.mul_(self.momentum).add_(g)
                g = g + self.momentum * v if self.nesterov else v
            p.sub_(self.lr * g)


class AdamScratch(_Base):
    """Adam (Kingma & Ba, 2014), and AdamW when `decoupled=True`.

        m = b1 * m + (1 - b1) * g              first moment: running mean of gradients   (direction/"momentum")
        v = b2 * v + (1 - b2) * g * g          second moment: running mean of g squared  (typical size)
        m_hat = m / (1 - b1**t)                bias correction: m and v start at 0, so early averages
        v_hat = v / (1 - b2**t)                are too small; this rescales them
        w -= lr * m_hat / (sqrt(v_hat) + eps)  big recent gradients -> smaller steps, tiny ones -> bigger

    Weight decay, two ways (the difference between Adam and AdamW):
      * Adam  (decoupled=False): add  wd * w  to the gradient BEFORE the moments. The decay then gets
                                 divided by sqrt(v_hat) like everything else, so weights with large
                                 gradients are regularised LESS. Not what we wanted.
      * AdamW (decoupled=True) : shrink the weights directly,  w *= 1 - lr * wd,  outside the adaptive
                                 machinery. Every weight decays at the same relative rate.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, decoupled: bool = False):
        super().__init__(params)
        self.lr, self.betas, self.eps, self.weight_decay, self.decoupled = lr, betas, eps, weight_decay, decoupled
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                if self.decoupled:
                    p.mul_(1.0 - self.lr * self.weight_decay)  # AdamW: decay the weights directly
                else:
                    g = g + self.weight_decay * p  # Adam: L2 term goes through the adaptive step
            m.mul_(b1).add_((1 - b1) * g)
            v.mul_(b2).add_((1 - b2) * g * g)
            m_hat = m / (1 - b1**self.t)
            v_hat = v / (1 - b2**self.t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
