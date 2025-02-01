import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


###############################################################################
#
# Building blocks for transformers
#
###############################################################################


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Remember the residual connection
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size


###############################################################################
#
# Transformer Decoder (we only use transformer decoder for our policies)
#
###############################################################################


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.
    """
    def __init__(self, dim, n_experts=8, n_expert_groups=1, n_limited_groups=1, 
                 n_activated_experts=2, score_func="softmax", route_scale=1.0):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        self.bias = None
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
            
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.
    """
    def __init__(self, input_size, mlp_hidden_size, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(input_size, mlp_hidden_size)
        self.w2 = nn.Linear(mlp_hidden_size, input_size)
        self.w3 = nn.Linear(input_size, mlp_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.
    """
    def __init__(self, input_size, mlp_hidden_size, n_experts=8, n_expert_groups=1, 
                 n_limited_groups=1, n_activated_experts=2, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts
        
        self.gate = Gate(
            input_size, 
            n_experts=n_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            n_activated_experts=n_activated_experts
        )
        
        self.experts = nn.ModuleList([
            Expert(input_size, mlp_hidden_size, dropout) 
            for _ in range(n_experts)
        ])

    def forward(self, x):
        original_shape = x.size()
        x = x.view(-1, self.input_size)
        
        # Get routing weights and expert indices
        weights, indices = self.gate(x)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Route inputs to experts
        for expert_idx in range(self.n_experts):
            idx, top = torch.where(indices == expert_idx)
            if len(idx) > 0:
                expert_output = self.experts[expert_idx](x[idx])
                output[idx] += expert_output * weights[idx, top, None]
        
        return output.view(original_shape)


class MoE_TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        n_dense_layers=1,  # Number of dense layers before MoE layers
        n_experts=8,       # Number of experts
        n_expert_groups=1,
        n_limited_groups=1,
        n_activated_experts=2,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.attention_output = {}
        
        self.n_dense_layers = n_dense_layers

        for layer_idx in range(num_layers):
            layer = nn.ModuleList([
                Norm(input_size),
                Attention(
                    input_size,
                    num_heads=num_heads,
                    head_output_size=head_output_size,
                    dropout=dropout,
                ),
                Norm(input_size),
            ])
            
            # Use dense layers for the first n_dense_layers, then MoE for the rest if use_moe is True
            if layer_idx < n_dense_layers:
                layer.append(
                    TransformerFeedForwardNN(
                        input_size, mlp_hidden_size, dropout=dropout
                    )
                )
            else:
                layer.append(
                    MoE(
                        input_size=input_size,
                        mlp_hidden_size=mlp_hidden_size,
                        n_experts=n_experts,
                        n_expert_groups=n_expert_groups,
                        n_limited_groups=n_limited_groups,
                        n_activated_experts=n_activated_experts,
                        dropout=dropout
                    )
                )
            
            self.layers.append(layer)
            self.attention_output[layer_idx] = None

        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                torch.triu(torch.ones(self.seq_len, self.seq_len))
                - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, layer in enumerate(self.layers):
            att_norm, att, ff_norm, ff = layer
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):

            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                torch.triu(torch.ones(self.seq_len, self.seq_len))
                - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device
