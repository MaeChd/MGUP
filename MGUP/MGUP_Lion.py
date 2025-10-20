from typing import Callable, Iterable, Tuple, Optional, Any, Dict
from torch.optim import Optimizer
import torch



def MGUP(update, grad, mask_ratio=0.5, alpha = 2.0, gamma=0.1):
    flat_values = (update * grad).flatten()
    k = int(mask_ratio * grad.numel())
    _, flat_indices = torch.topk(flat_values, k)
    mask = torch.full_like(flat_values, gamma)
    mask[flat_indices] = alpha
    mask = mask.reshape_as(grad)
    return mask



def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, mask_ratio, alpha, gamma):
    # stepweight decay
    p.add_(p, alpha=-lr * wd)
    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    mask = MGUP(update, grad, mask_ratio,alpha, gamma)
    p.add_(update * mask, alpha=-lr)
    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


class Lion(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
            mask_ratio=0.5,
            alpha=2.0,
            gamma=0.1
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            mask_ratio=mask_ratio,
            alpha=alpha,
            gamma=gamma
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], \
                    self.state[p]
                mask_ratio = group['mask_ratio']
                alpha, gamma = group['alpha'], group['gamma']
                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    mask_ratio,
                    alpha,
                    gamma
                )

        return loss

