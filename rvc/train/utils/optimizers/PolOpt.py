import math

import torch
from torch.optim.optimizer import Optimizer


class PolOpt(Optimizer):
    """PolOpt (AdaBelief / Yogi Hybrid Optimizer)

    1. Yogi-sign control для второго момента: вместо резкого изменения дисперсии (variance),
       используется знак разности sign((g - m)^2 - v), что предотвращает падение знаменателя в ноль
       и взрывные скачки learning rate при игре генератора и дискриминатора.
    2. Раздельный (Decoupled) Weight Decay как в AdamW для улучшения генерализации.
    3. Корректная позиция Epsilon (снаружи корня после коррекции смещения).
    4. Trust Region Clamping (max_step_clip): ограничение максимального шага обновления одного параметра
       для защиты структуры акустических фильтров при градиентных всплесках.

    Arguments:
        params: параметры для оптимизации
        lr: learning rate (по умолчанию: 1e-4)
        betas: коэффициенты сглаживания (β₁, β₂) (по умолчанию: (0.8, 0.99))
        eps: член числовой стабильности (по умолчанию: 1e-7)
        weight_decay: decoupled weight decay (по умолчанию: 0.01)
        max_step_clip: максимальная абсолютная величина шага обновления (по умолчанию: 1.0)

    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-7,
        weight_decay: float = 0.01,
        max_step_clip: float = 1.0,
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
        if max_step_clip < 0:
            raise ValueError(f"Invalid max_step_clip value: {max_step_clip}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_step_clip=max_step_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            max_clip = group["max_step_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PolOpt does not support sparse gradients.")

                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                # 1. Decoupled Weight Decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # 2. Обновление скользящего среднего градиента (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. Вычисление ошибки "веры" (Belief residual): grad - exp_avg
                grad_res = grad - exp_avg
                grad_res_sq = grad_res * grad_res

                # 4. Yogi-Belief гибридное обновление второго момента (v_t)
                # Контролирует скорость изменения дисперсии через знак разности, не давая знаменателю схлопнуться
                diff = grad_res_sq - exp_avg_var
                exp_avg_var.addcmul_(torch.sign(diff), grad_res_sq, value=1.0 - beta2)

                # 5. Коррекция смещения (Bias correction)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1

                # 6. Знаменатель с безопасным размещением eps СНАРУЖИ корня
                denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 7. Нормированный вектор обновления весов
                update = exp_avg / denom

                # 8. Trust Region Clamping: защита акустических фильтров от градиентных всплесков
                if max_clip > 0:
                    update = torch.clamp(update, -max_clip, max_clip)

                # 9. Применение обновления к параметрам
                p.add_(update, alpha=-step_size)

        return loss
