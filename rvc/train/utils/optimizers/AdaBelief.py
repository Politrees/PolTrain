from typing import Dict, List, Tuple
import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ("AdaBelief",)


class AdaBelief(Optimizer):
    """AdaBelief Optimizer с поддержкой Gradient Centralization.

    Исправленная версия без багов оригинала + GC для стабильности GAN.

    Arguments:
        params: parameters to optimize
        lr: learning rate (default: 1e-4)
        betas: coefficients (β₁, β₂) (default: (0.8, 0.99))
        eps: numerical stability (default: 1e-10)
        weight_decay: decoupled weight decay (default: 0)
        foreach: use torch._foreach for speed (default: True)
        use_gc: enable Gradient Centralization (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-10,
        weight_decay: float = 0,
        foreach: bool = True,
        use_gc: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            use_gc=use_gc,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("use_gc", True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["foreach"]:
                self._step_foreach(group)
            else:
                self._step_single(group)

        return loss

    def _step_single(self, group: Dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        use_gc = group["use_gc"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdaBelief does not support sparse gradients")

            # Gradient Centralization
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_var = state["exp_avg_var"]

            state["step"] += 1
            step = state["step"]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            # m_t = β₁·m_{t-1} + (1-β₁)·g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # s_t = β₂·s_{t-1} + (1-β₂)·(g_t - m_t)²
            grad_residual = grad - exp_avg
            exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

            # denom = sqrt(s_t + eps) / sqrt(bc2)
            denom = (exp_avg_var.add(eps)).sqrt() / math.sqrt(bias_correction2)

            step_size = lr / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_foreach(self, group: Dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        use_gc = group["use_gc"]

        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_vars: List[Tensor] = []

        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grad = p.grad
            
            # Gradient Centralization (Foreach)
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
            
            grads.append(grad)

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_vars.append(state["exp_avg_var"])

        if not params_with_grad:
            return

        state = self.state[params_with_grad[0]]
        state["step"] += 1
        step = state["step"]
        for p in params_with_grad[1:]:
            self.state[p]["step"] = step

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        # grad_residuals = grad - exp_avg
        grad_residuals = torch._foreach_sub(grads, exp_avgs)
        torch._foreach_mul_(exp_avg_vars, beta2)
        torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1 - beta2)

        denom = torch._foreach_add(exp_avg_vars, eps)
        denom = torch._foreach_sqrt(denom)
        torch._foreach_div_(denom, math.sqrt(bias_correction2))

        step_size = lr / bias_correction1
        updates = torch._foreach_div(exp_avgs, denom)
        torch._foreach_add_(params_with_grad, updates, alpha=-step_size)
