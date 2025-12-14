"""
Here is an original implementation of SpAdamW.
"""

import torch
import math
import numpy as np

from .muon import zeropower_via_newtonschulz5
from .ademamix import linear_warmup_scheduler, linear_hl_warmup_scheduler

@torch.compile
def matrix_inv_sqrt_NS(A: torch.Tensor, alpha: torch.Tensor, ns_iter: int=10):
    assert A.ndim == 2 and A.size(0) == A.size(1), "A must be square"
    device = A.device
    n = A.size(0)
    # scaled matrix
    Ahat = A / alpha
    # init
    Y = Ahat.clone()
    I = torch.eye(n, dtype=A.dtype, device=device)
    Z = I.clone()
    # Newton–Schulz iterations
    for _ in range(ns_iter):
        T = 0.5 * (3.0 * I - Z @ Y)   
        Y = Y @ T
        Z = T @ Z
    # Rescale Z to get A^{-1/2}
    return Z / alpha.sqrt()

def clip_sigvals(G: torch.Tensor, clip_c: float=1.0, ns_iter: int = 10):
    X = G.bfloat16()
    # X = G.float()
    #clip_c = torch.as_tensor(clip_c, dtype=G.dtype, device=G.device)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    XXT = X @ X.mT 
    s_max = torch.minimum(X.norm(dim=(-2, -1)), 
                      XXT.abs().sum(dim=-2).max().sqrt()) + 1e-7
    if s_max <= clip_c:
        return G 
    InvSqrt = matrix_inv_sqrt_NS(
        (XXT / clip_c**2) + torch.eye(X.size(-2), dtype=X.dtype, device=X.device),
        alpha=(s_max / clip_c)**2 + 1,
        ns_iter=ns_iter
    )
    out = InvSqrt @ X
    if transposed:
        out = out.mT
    return out

class ClippingSchedule:
    def __init__(self, c0, c1, warmup_iters, mode="cos", k=10.0):
        """ Clipping schedule from c0 to c1 over warmup_iters iterations. 
            Then it stays at c1.
        Modes: 'cos' (cosine), 'exp' (exponential), 'linear' (linear).
        """

        self.c0 = float(c0)   # start threshold
        self.c1 = float(c1)   # end threshold
        self.T = int(warmup_iters)
        self.mode = mode
        self.k = float(k)     # decay rate for exp

    def __call__(self, step: int) -> float:
        # if we’re past training, just freeze at c1
        if step >= self.T:
            return self.c1
        # normalized progress [0,1]
        t = step / self.T
        # schedule in [0,1], decreasing
        if self.mode == "cos":
            s = 0.5 * (1 + math.cos(math.pi * t))   # cosine
        elif self.mode == "exp":
            s = math.exp(-self.k * t)               # fast decay
        elif self.mode == "linear":
            s = 1.0 - t                             # straight line
        else:
            raise ValueError("Unknown mode: choose 'cos', 'exp', or 'linear'")
        # interpolate between c0 and c1
        return self.c1 + (self.c0 - self.c1) * s


class SpAdamW(torch.optim.Optimizer):
    """ Implements AdamW algorithm with spectral clipping / normalization.
    Parameters:
        lr (float): learning rate for the matrix update after clipping. Default 1e-3.
        betas (tuple of 2 floats): adam's beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-8
        weight_decay (float): Weight decay. Default: 0.1
        clip_c (tuple): spectral clipping parameter (start and end). Default (10,10).
        sp_clip_mode (str): spectral clipping schedule mode: 'cos', 'exp', or 'linear'. Default 'cos'.
        ns_iter (int): number of Newton-Schulz iterations for spectral clipping. Default 10.
        warmup_iters (int): total number of warm-up iterations for clipping. Default 2000.
        normalization (bool): if True, use spectral normalization instead of clipping. Default False.
    """

    def __init__(self, 
                params, 
                lr=1e-3, 
                betas=(0.9, 0.999), 
                eps=1e-8, 
                weight_decay=0.1, 
                clip_c=(10,10),
                sp_clip_mode='cos',
                ns_iter=10,
                warmup_iters=2000,
                bias_correction=True,
                normalization=False
            ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if not (clip_c[0] > 0.0 and clip_c[1] > 0.0):
            raise ValueError("Invalid clip_c parameters: {} - should be c0,c1 > 0.0".format(clip_c))
        if not sp_clip_mode in ['cos', 'exp', 'linear']:
            raise ValueError("Invalid spectral clipping mode: {} - should be 'cos', 'exp', or 'linear'".format(sp_clip_mode))
        if not warmup_iters >= 0:
            raise ValueError("Invalid warmup_iters for clipping: {} - should be >= 0".format(warmup_iters))
        if not ns_iter >= 1:
            raise ValueError("Invalid ns_iter: {} - should be >= 1".format(ns_iter))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {} - should be >= 0.0".format(weight_decay))
        if not isinstance(normalization, bool):
            raise ValueError("Invalid normalization value: {} - should be boolean".format(normalization))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        ns_iter=ns_iter, bias_correction=bias_correction, normalization=normalization)
        super().__init__(params, defaults)
        self.clipping_schedule = ClippingSchedule(clip_c[0], clip_c[1], warmup_iters, mode=sp_clip_mode)

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

        with torch.no_grad():
            state_full = self.state 
            if state_full.get('step') is None:
                state_full['step'] = 0
            clip_c = self.clipping_schedule(state_full["step"])
            #wandb.log({"train/clip_c": clip_c}, step=state_full["step"]+1)

            for group in self.param_groups:
                step_size = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                beta1, beta2 = group["betas"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    state = self.state[p]
                    
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    if group["bias_correction"]:
                        bias_correction1 = 1 - beta1 ** state["step"]
                        bias_correction2 = 1 - beta2 ** state["step"]
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                        step_num = exp_avg / bias_correction1
                        update = step_num / denom
                    else:
                        update = exp_avg / (exp_avg_sq.sqrt() + eps)
                    
                    if 2 >= state["exp_avg"].dim() > 1:
                        if group['normalization']:
                            update = zeropower_via_newtonschulz5(update, 5)
                        else:
                            update = clip_sigvals(update, clip_c, group['ns_iter'])
                        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
                        p.data.add_(-step_size * update)

                    elif state["exp_avg"].dim() == 1:   
                        vec_sinval = update.norm()
                        if group['normalization']:
                            update /= (vec_sinval + eps)
                        else:
                            if vec_sinval > clip_c:
                                update *= (clip_c / vec_sinval)
                        p.data.add_(-step_size * update)
                    else:
                        NotImplementedError("Only implemented methods for parameters with 1 or 2 dimensions.")

                    if group["weight_decay"] > 0.0:
                        p.data.add_(p.data, alpha=-step_size * weight_decay)    
                   
            state_full["step"] += 1
        return loss
    

class SpAdEMAMix(torch.optim.Optimizer):
    r"""Implements the SpAdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999))
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
        clip_c (float): spectral clipping parameter. Default 1.
        sp_clip_mode (str): spectral clipping schedule mode: 'cos', 'exp', or 'linear'. Default 'cos'.
        ns_iter (int): number of Newton-Schulz iterations for spectral clipping. Default 10.
        normalization (bool): if True, use spectral normalization instead of clipping. Default False.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=2.0,
        beta3_warmup=None,
        alpha_warmup=None,
        eps=1e-8,
        weight_decay=0,
        clip_c=1,
        sp_clip_mode='cos',
        ns_iter=10,
        normalization=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
            clip_c=clip_c, 
            ns_iter=ns_iter, 
            normalization=normalization
        )
        super(SpAdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SpAdEMAMix, self).__setstate__(state)

    @torch.no_grad()
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
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]
            clip_c = group["clip_c"]
            ns_iter = group["ns_iter"]
            normalization = group["normalization"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg_fast, exp_avg_slow, exp_avg_sq = (
                    state["exp_avg_fast"],
                    state["exp_avg_slow"],
                    state["exp_avg_sq"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute the effective alpha and beta3 in case warmup is used
                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        state["step"],
                        alpha_end=alpha_final,
                        alpha_start=0,
                        warmup=alpha_warmup,
                    )
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=beta3_final,
                        beta_start=beta1,
                        warmup=beta3_warmup,
                    )
                else:
                    beta3 = beta3_final

                # Decay the first and second moment running average coefficient
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = (
                    exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow
                ) / denom

                if 2 >= update.dim() > 1:
                    if normalization:
                        update = zeropower_via_newtonschulz5(update, 5)
                    else:
                        update = clip_sigvals(update, clip_c, ns_iter)
                    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
                    p.data.add_(-lr*update)

                elif update.dim() == 1:   
                    vec_sinval = update.norm()
                    if group['normalization']:
                        update /= (vec_sinval + 1e-7)
                    else:
                        if vec_sinval > clip_c:
                            update *= (clip_c / vec_sinval)
                    p.data.add_(-lr*update)

                else:
                    NotImplementedError("Only implemented methods for parameters with 1 or 2 dimensions.")

                # decay
                if lmbda > 0.0:
                    p.data.add_(p.data, alpha=-lr * lmbda)

        return loss


