import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=bias
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv_proj(x)

        queries, keys, values = qkv.chunk(3, dim=-1)

        queries = queries.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        keys = keys.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        values = values.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(
                T,
                T,
                device=x.device,
                dtype=torch.bool
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_weights = self.attn_dropout(
            attention_weights
        )

        context = attention_weights @ values

        context = context.transpose(1, 2)

        context = context.contiguous().view(
            B,
            T,
            D
        )

        output = self.out_proj(context)

        return output