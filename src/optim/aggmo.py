"""
Here is an original implementation of AggMo.
Source: https://github.com/AtheMathmo/AggMo/
"""

import torch
import math

class AggMo(torch.optim.Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent"""

    def __init__(
            self, 
            params, 
            lr=1e-3, 
            betas=[0.0, 0.9, 0.99], 
            weight_decay=0,
            decouple=True,
        ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if len(betas) == 0:
            raise ValueError("Invalid list of betas: betas should contain at least one momentum term")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError("Invalid beta value at index {}: {}".format(i, beta))
        defaults = dict(
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay,
            decouple=decouple
        )
        super(AggMo, self).__init__(params, defaults)

    @classmethod
    def from_exp_form(cls, params, lr=1e-3, a=0.1, k=3, weight_decay=0, decouple=True):
        betas = [1 - a**i for i in range(k)]
        return cls(params, lr, betas, weight_decay, decouple)

    def __setstate__(self, state):
        super(AggMo, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            decouple = group["decouple"]
            total_mom = float(len(betas))

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    if decouple:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = {}
                    for beta in betas:
                        param_state["momentum_buffer"][beta] = torch.zeros_like(p.data)
                for beta in betas:
                    buf = param_state["momentum_buffer"][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.add_(buf, alpha=-lr / total_mom)
        return loss

    def zero_momentum_buffers(self):
        for group in self.param_groups:
            betas = group["betas"]
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum_buffer"] = {}
                for beta in betas:
                    param_state["momentum_buffer"][beta] = torch.zeros_like(p.data)

    def update_hparam(self, name, value):
        for param_group in self.param_groups:
            param_group[name] = value


class AggMo2(torch.optim.Optimizer):
    r"""
    AggMo2: Aggregated Momentum with Adaptive Scaling.
    Combines K momentum buffers rescaled by an EMA of the second moment estimate.
    """

    def __init__(
            self, 
            params, 
            lr=1e-3, 
            betas=[0.0, 0.9, 0.99], 
            beta2=0.999,
            eps=1e-8,
            weight_decay=0,
            decouple=True,
        ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError("Invalid beta value at index {}: {}".format(i, beta))
                
        defaults = dict(
            lr=lr, 
            betas=betas, 
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple
        )
        super(AggMo2, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            beta2 = group["beta2"]
            eps = group["eps"]
            decouple = group["decouple"]
            total_mom = float(len(betas))

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["momentum_buffers"] = [torch.zeros_like(p.data) for _ in betas]

                exp_avg_sq = state["exp_avg_sq"]
                mom_buffers = state["momentum_buffers"]
                state["step"] += 1

                if weight_decay != 0:
                    if decouple:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(p.data, alpha=weight_decay)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update_acc = torch.zeros_like(p.data)
                
                for i, beta in enumerate(betas):
                    buf = mom_buffers[i]
                    buf.mul_(beta).add_(grad)
                    update_acc.add_(buf, alpha=1.0 / total_mom)

                p.data.addcdiv_(update_acc, denom, value=-lr)

        return loss