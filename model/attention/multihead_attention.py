import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        attention_type: str = "standard",
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        if attention_type not in {"standard", "flash"}:
            raise ValueError(
                "attention_type must be either 'standard' or 'flash'."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.attention_type = attention_type

        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=bias,
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (B, T, D), got {tuple(x.shape)}"
            )

        B, T, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected hidden size {self.d_model}, got {D}"
            )

        qkv = self.qkv_proj(x)

        queries, keys, values = qkv.chunk(
            3,
            dim=-1,
        )

        queries = queries.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        keys = keys.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        values = values.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        past_length = 0

        if past_kv is not None:
            past_keys, past_values = past_kv
            past_length = past_keys.size(2)

            keys = torch.cat(
                [past_keys, keys],
                dim=2,
            )

            values = torch.cat(
                [past_values, values],
                dim=2,
            )

        present_kv = (
            keys,
            values,
        ) if use_cache else None

        total_length = past_length + T

        query_positions = torch.arange(
            past_length,
            total_length,
            device=x.device,
        ).unsqueeze(-1)

        key_positions = torch.arange(
            total_length,
            device=x.device,
        ).unsqueeze(0)

        causal_mask = key_positions <= query_positions
        causal_mask = causal_mask.view(1, 1, T, total_length)

        if self.attention_type == "flash":
            dropout_p = self.dropout if self.training else 0.0

            context = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        else:
            scores = queries @ keys.transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)

            scores = scores.masked_fill(
                ~causal_mask,
                float("-inf"),
            )

            attention_weights = torch.softmax(
                scores,
                dim=-1,
            )

            attention_weights = self.attn_dropout(
                attention_weights
            )

            context = attention_weights @ values

        context = context.transpose(1, 2)

        context = context.contiguous().view(
            B,
            T,
            D,
        )

        output = self.out_proj(context)

        if use_cache:
            return output, present_kv

        return output