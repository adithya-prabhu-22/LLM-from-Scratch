import math
import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model

        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj = nn.Linear(d_model, d_model, bias=bias)
        self.value_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(D)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        context = attention_weights @ values
        output = self.out_proj(context)

        return output