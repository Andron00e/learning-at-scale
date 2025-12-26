"""
Here is an original implementation of Unit Optimizers. 
Projects all weights matrices to have unit norms along the embedding dimension:
nn.Linear: (out_dim in_dim), we normalize each out_dim.
nn.Embedding: (num_embeddings, embedding_dim), we normalize each embedding.
In summary, compute the mean across the second dim.
"""

import math
from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
import torch
from typing import Dict, Tuple, Union, Optional, Iterable, Any, Callable
from typing_extensions import TypeAlias
from itertools import chain
import torch.distributed as dist
from .muon import MuonDistMeta, adjust_lr_wd_for_muon, normalize_range, zeropower_via_newtonschulz5
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class UnitAdam(torch.optim.Optimizer):
    """
    Implements Adam with Unit Norm Projection.
    Weight decay is omitted.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        lr_scale=1.0,
        weight_decay=None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay is not None:
            print(f"Warning: UnitAdam ignored {weight_decay=}")
        defaults = dict(lr=lr, betas=betas, eps=eps, lr_scale=lr_scale)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

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
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                step_size = group["lr_scale"] * group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if p.dim() == 2:
                    # Normalize "along the embedding" dimension
                    p.div_(p.square().sum(dim=1, keepdim=True).sqrt())

        return loss


class UnitAdEMAMix(torch.optim.Optimizer):
    """
    Implements AdEMAMix with Unit Norm Projection.
    Weight decay is omitted.
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
        lr_scale=1.0,
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
        if weight_decay != 0:
            print(f"Warning: UnitAdEMAMix ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")
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
            lr_scale=lr_scale,
        )
        super(UnitAdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UnitAdEMAMix, self).__setstate__(state)

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
            lr = group["lr"] * group["lr_scale"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

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

                # decay
                update.add_(p, alpha=lmbda)

                p.add_(-lr * update)

                if p.dim() == 2:
                    p.div_(p.square().sum(dim=1, keepdim=True).sqrt())

        return loss


class UnitSGD(torch.optim.Optimizer):
    """
    Implements SGD with Momentum and Unit Norm Projection (Projected SGD).
    If sign_update=True, this becomes UnitSignSGD.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        sign_update=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay != 0:
            print(f"Warning: UnitSGD ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign_update=sign_update,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("sign_update", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            sign_update = group["sign_update"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.clone(grad).detach()
                
                state["step"] += 1
                
                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        update_grad = grad.add(buf, alpha=momentum)
                    else:
                        update_grad = buf
                else:
                    update_grad = grad

                if sign_update:
                    update_grad = update_grad.sign()

                p.add_(update_grad, alpha=-lr)

                if p.dim() == 2:
                    p.div_(p.square().sum(dim=1, keepdim=True).sqrt())

        return loss
    

class UnitDistributedMuon(torch.optim.Optimizer):
    """
    Implements Kimi Muon with Unit Norm Projection.
    Weight decay is omitted.
    """
    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        lr_scale=1.0
    ):
        if weight_decay != 0:
            print(f"Warning: UnitDistributedMuon ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            lr_scale=lr_scale,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group["params"]:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                if p.ndim >= 2 and p.size(0) < 10000:
                    self.state[p]["use_muon"] = True
                else:
                    self.state[p]["use_muon"] = False

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        dist_group,
        tp_group,
        dist_metas: Dict[torch.nn.Parameter, MuonDistMeta],
    ):
        """
        enable distributed mode
        Args:
            global_buffer_size: global buffer size
            dist group: optimizer sharding group
            tp group: param tp group
            dist metas: dist metas for all param
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (
                    local_buffer_start,
                    local_buffer_start + local_buffer_size,
                )
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def step(self):

        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # update muon momentum first
        for group in self.param_groups:

            momentum = group["momentum"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                # 1-dim grad for distributed mode
                assert self.distributed_mode or g.dim() == 2

                # prepare muon buffer in state
                state = self.state[p]
                if not "muon_buffer" in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)

                # save to ns input
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_inputs[p] = g.bfloat16()

        # rewrite ns_inputs if distributed
        if self.distributed_mode:

            # initialize buffers
            ns_input_local_buffers = [
                [
                    torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes
                ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [
                    torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes
                ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_range = normalize_range(
                    dist_meta.local_range, local_buffer_range[0]
                )
                ns_input_local_buffer[local_range[0] : local_range[1]].copy_(
                    ns_input.view(-1)
                )

            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(
                ns_input_global_buffers, ns_input_local_buffers
            ):
                for ns_input_global_bucket, ns_input_local_bucket in zip(
                    ns_input_global_buffer, ns_input_local_buffer
                ):
                    dist.all_gather_into_tensor(
                        ns_input_global_bucket,
                        ns_input_local_bucket,
                        group=self.dist_group,
                    )

            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # update muon momentum first
        for group in self.param_groups:

            # if not group.get('use_muon', False):
            #     continue

            lr = group["lr"] * group["lr_scale"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:

                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    ns_input_shards = [
                        torch.empty_like(ns_input) for _ in range(tp_world_size)
                    ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                # only local buffer part
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    update = update.reshape(-1)[
                        local_range_in_global_range[0] : local_range_in_global_range[1]
                    ]

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)

                #  adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            # if group.get('use_muon', False):
            #     continue

            # init step
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"] * group["lr_scale"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:

                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        # Normalize "along the embedding" dimension
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() == 2:
                    p.data.div_(p.data.square().sum(dim=1, keepdim=True).sqrt())


class UnitSOAP(torch.optim.Optimizer):
    """
    Implements SOAP with Unit Norm Projection.
    Weight decay is omitted.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0,
        lr_scale: float = 1.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,  #
        merge_dims: bool = False,  # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        if weight_decay != 0:
            print(f"Warning: UnitSOAP ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
            "lr_scale": lr_scale,
        }
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(
                    grad_projected.square(), alpha=(1.0 - beta2)
                )

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = self.project(
                    exp_avg,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                step_size = group["lr"] * group["lr_scale"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad**2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                # From AdamW code: Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["lr_scale"] * group["weight_decay"]))

                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

                if p.dim() == 2:
                    # Normalize "along the embedding" dimension
                    p.div_(p.square().sum(dim=1, keepdim=True).sqrt())

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state["GG"] = (
            []
        )  # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None  # Will hold all the eigenbases of the preconditioner.
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [0]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
    ):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx), range(idx + 1, len(new_grad.shape))
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for k.
                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(
                state, max_precond_dim, merge_dims
            )

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [1]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(
                    m + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
            except:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state["GG"]
        orth_list = state["Q"]

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return final


class UnitProdigy(torch.optim.Optimizer):
    """
    Implements Prodigy with Unit Norm Projection.
    Weight decay is omitted.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        beta3=None,
        eps=1e-8,
        weight_decay=0,
        lr_scale=1.0,
        decouple=True,
        use_bias_correction=False,
        safeguard_warmup=False,
        d0=1e-6,
        d_coef=1.0,
        growth_rate=float("inf"),
        fsdp_in_use=False,
    ):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay != 0:
            print(f"Warning: UnitProdigy ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")
        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            d=d0,
            d0=d0,
            d_max=d0,
            d_numerator=0.0,
            d_coef=d_coef,
            k=0,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            safeguard_warmup=safeguard_warmup,
            fsdp_in_use=fsdp_in_use,
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group["use_bias_correction"]
        beta1, beta2 = group["betas"]
        beta3 = group["beta3"]
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group["k"]

        d = group["d"]
        d_max = group["d_max"]
        d_coef = group["d_coef"]
        lr = max(group["lr"] * group["lr_scale"] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1

        # we will compute dlr on a per-group basis
        # dlr = d * lr * bias_correction

        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        fsdp_in_use = group["fsdp_in_use"]

        d_numerator = group["d_numerator"]
        d_numerator *= beta3

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]
            group_lr = group["lr"] * group["lr_scale"]
            d0 = group["d0"]
            safeguard_warmup = group["safeguard_warmup"]

            # remove this check for nGPT
            # if group_lr not in [lr, 0.0]:
            #     raise RuntimeError(
            #         f"Setting different lr values in different parameter groups is only supported for values of 0"
            #     )

            dlr = d * group_lr * bias_correction # compute dlr for this group

            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data).detach()
                    state["p0"] = p.detach().clone()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                s = state["s"]
                p0 = state["p0"]

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    d_numerator += (
                        (d / d0)
                        * dlr
                        * torch.dot(grad.flatten(), (p0.data - p.data).flatten()).item()
                    )

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=d * d * (1 - beta2)
                    )

                    if safeguard_warmup:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

            ######

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = global_d_denom
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat

            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            group_lr = group["lr"] * group["lr_scale"]
            dlr = d * group_lr * bias_correction  # re-compute dlr for this group

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-dlr)

                if p.dim() == 2:
                    # Normalize "along the embedding" dimension
                    p.data.div_(p.data.square().sum(dim=1, keepdim=True).sqrt())

            group["k"] = k + 1

        return loss


class UnitAdamWScheduleFree(torch.optim.Optimizer):
    """
    Implements AdamW with a specific learning rate schedule and Unit Norm Projection.
    Weight decay is omitted but can be applied at y (the exponentially weighted average of weights).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 0.0025,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        warmup_steps: int = 0,
        lr_scale: float = 1.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
    ):
        if weight_decay != 0:
            print(f"Warning: UnitAdamWScheduleFree ignores weight_decay (set to {weight_decay}) due to Unit Norm constraint.")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to y
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * group["lr_scale"] * sched
            group["scheduled_lr"] = lr  # For logging purposes

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(
                        p, memory_format=torch.preserve_format
                    )
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            if group["foreach"] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(
                    *[
                        (p, p.grad, self.state[p]["exp_avg_sq"], self.state[p]["z"])
                        for p in active_p
                    ]
                )

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p  # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state["z"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            # project both y and z sequences
            for p in active_p:
                if p.dim() == 2:
                    # Normalize the active param y "along the embedding" dimension
                    p.div_(p.square().sum(dim=1, keepdim=True).sqrt())
                    
                    # if we dont do this, the next .lerp() will pull p off the sphere
                    z = self.state[p]['z']
                    z.div_(z.square().sum(dim=1, keepdim=True).sqrt())

            group["k"] = k + 1
        return loss