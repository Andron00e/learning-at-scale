"""
Full definition of a nGPT Language Model, all of it in this single file.
The changes to the normal GPT-2 model are marked with 'ngpt change here!'.
References:
1) the official but simplified nGPT PyTorch implementation released by NVIDIA:
https://github.com/NVIDIA/ngpt/blob/main/model.py
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.llama import precompute_freqs_cis, apply_rotary_emb
from models.base import CausalSelfAttention, GPTBase
from models.moe import MoE


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


class EmbeddingNorm(nn.Module): # ngpt change here!
    def __init__(self, dim: int = None, eps: float = 1e-6, rms: bool = False, gain: bool = False):
        super().__init__()
        self.eps = eps
        self.rms = rms
        self.has_gain = gain
        if self.has_gain:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        if self.rms:
            # Make the rms of each token equal to one
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            # Make the norm of each token equal to one
            return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.has_gain:
            output = output * self.weight
        return output


class NormalizedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.n_embd = config.n_embd # ngpt change here!
        self.w_u = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.s_u = nn.Parameter(torch.ones(hidden_dim)) # ngpt change here!
        self.w_nu = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.s_nu = nn.Parameter(torch.ones(hidden_dim)) # ngpt change here!
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # tuple form because of aux loss from MoE
        u = self.w_u(x) * self.s_u
        nu = self.w_nu(x) * self.s_nu * math.sqrt(self.n_embd)
        return self.c_proj(nn.functional.silu(nu) * u), {}
    

class NormalizedAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.q_norm = EmbeddingNorm()
        self.k_norm = EmbeddingNorm()
        self.s_qk = nn.Parameter(torch.ones(config.n_embd))
    
    def forward(self, x, freqs_cis):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (B, nh, T, hs)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # nGPT apply token normalization to q, k and scale with the same gain
        # Note that this is done after the rotary embeddings
        q = self.q_norm(q) * self.s_qk.view(1, self.n_head, 1, C // self.n_head) # ngpt change here!
        k = self.k_norm(k) * self.s_qk.view(1, self.n_head, 1, C // self.n_head) # ngpt change here!

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True,
                scale=math.sqrt(k.size(-1))
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * math.sqrt(k.size(-1)) # ngpt change here!
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class NormalizedBlock(nn.Module): # ngpt change here!
    def __init__(self, config):
        super().__init__()
        self.attn = NormalizedAttention(config)
        self.attn_out_norm = EmbeddingNorm()
        self.attn_add_norm = EmbeddingNorm()
        self.alpha_a = nn.Parameter(torch.full((config.n_embd,), 0.05))

        if config.moe:
            self.mlp = MoE(config, NormalizedMLP)
        else:
            self.mlp = NormalizedMLP(config)
        
        self.mlp_out_norm = EmbeddingNorm()
        self.mlp_add_norm = EmbeddingNorm()
        self.alpha_m = nn.Parameter(torch.full((config.n_embd,), 0.05))

    def forward(self, x, freqs_cis):
        attn_out = self.attn_out_norm(self.attn(x, freqs_cis))
        x = (1 - self.alpha_a) * x + self.alpha_a * attn_out
        x = self.attn_add_norm(x)
        
        x_, logits_and_experts = self.mlp(x)
        # we do not normalize MoE logits
        x_ = self.mlp_out_norm(x_)
        x = (1 - self.alpha_m) * x + self.alpha_m * x_
        x = self.mlp_add_norm(x)
        return x, logits_and_experts
    

class NormalizedGPT(GPTBase): # ngpt change here!
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([NormalizedBlock(config) for _ in range(config.n_layer)]),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if not config.untied_embeds:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        self.s_z = nn.Parameter(torch.full((config.vocab_size,), config.s_z_init)) # ngpt change here!

        # init all weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("router.weight"):
                # special scaled init to moe router?
                # i am not sure if this is needed in nGPT, but keeping it just in case
                with torch.no_grad():
                    std = p.std()
                    p.div_(p.sum(dim=1, keepdim=True))
                    p.mul_(std / p.std())

    @torch.no_grad()
    def _init_weights(self, module): # ngpt change here!
        if isinstance(module, nn.Linear):
            # out_dim in_dim
            torch.nn.init.normal_(module.weight, mean=0.0)
            module.weight.div_(module.weight.square().sum(dim=1, keepdim=True).sqrt())
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # num_embeddings, embedding_dim
            torch.nn.init.normal_(module.weight, mean=0.0)
            module.weight.div_(module.weight.square().sum(dim=1, keepdim=True).sqrt())
        # Keep the custom init of the gains in NormalizedBlock (0.05), NormalizedMLP (1), EmbeddingNorm (1), NormalizedGPT (1)
    
    def get_parameter_group_specs(self, config): # ngpt change here!
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        accelerated_gains_1 = []
        accelerated_gains_2 = []
        standard_params = []

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False): # only look at direct params
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn in ["s_qk", "s_z"]:
                    # These gains use a higher learning rate of sqrt(dim)
                    accelerated_gains_1.append(p)
                elif pn in ["alpha_a", "alpha_m"]:
                    # These gains use a higher learning rate of 0.05 * sqrt(dim)
                    accelerated_gains_2.append(p)
                elif pn in ["s_u", "s_nu"]:
                    # These gains use an unmodified learning rate!
                    standard_params.append(p)
                elif "wte" in fpn and not self.config.no_weight_tying:
                    # Don't add this if we are doing weight tying
                    pass
                else:
                    # Embeddings, standard weight matrices use the standard rate
                    standard_params.append(p)

        # validate that we considered every parameter
        all_params = set(self.parameters(recurse=True))
        assigned_params = set(accelerated_gains_1 + accelerated_gains_2 + standard_params)
        assert (
            all_params == assigned_params
        ), "every parameter in the model must be assigned to exactly one optimizer group"
        assert (
            len(assigned_params) == len(accelerated_gains_1) + len(accelerated_gains_2) + len(standard_params)
        ), "parameters should not be duplicated across groups"

        # create the pytorch optimizer object
        return [
            {"params": standard_params},
            {"params": accelerated_gains_1, "lr_scale": self.config.n_embd ** 0.5},
            {"params": accelerated_gains_2, "lr_scale": 0.05 * self.config.n_embd ** 0.5},
        ]
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False, moe=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        # router logits is a list for each layer's routing, each of shape (b * seq_len, n_experts)
        router_logits = []
        # experts is a list for each layer's selected experts, shape (b * seq_len, topk)
        experts = []

        for block in self.transformer.h:
            x, logits_and_experts = block(x, freqs_cis=freqs_cis)
            if len(logits_and_experts) > 0:
                router_logits.append(logits_and_experts["router_logits"])
                experts.append(logits_and_experts["selected_experts"])

        # aux_losses is a dict with keys for different auxiliary losses
        aux_losses = {}
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) * self.s_z # ngpt change here!
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            if moe and self.config.moe_routing == "standard_gating":
                # calculate the router losses per layer
                for logit, expert_choice in zip(router_logits, experts):
                    router_losses = self.get_router_losses(
                        logit, expert_choice, eval=not self.training
                    )
                    for k, v in router_losses.items():
                        aux_losses[k] = aux_losses.get(k, 0.0) + v
                        if self.training:
                            loss += (
                                v
                                * getattr(self.config, k + "_factor")
                                / self.config.n_layer
                            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits * self.s_z # ngpt change here!
            loss = None

        logits = logits if get_logits else None

        router_logits = (
            torch.stack(router_logits, dim=0) if len(router_logits) > 0 else None
        )

        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "router_logits": router_logits,
        }
