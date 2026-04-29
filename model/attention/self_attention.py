import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()

        self.d_model = d_model

        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj = nn.Linear(d_model, d_model, bias=bias)
        self.value_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(D)

        attention_weights = torch.softmax(scores, dim=-1)

        context = attention_weights @ values

        return context