import torch
import torch.nn as nn

from model.layer_norm import LayerNorm
from model.attention.multihead_attention import MultiHeadAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        ffn_hidden_dim: int | None = None,
        attention_type: str = "standard",
    ):
        super().__init__()

        self.ln_1 = LayerNorm(
            d_model=d_model,
            bias=bias,
        )

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            attention_type=attention_type,
        )

        self.ln_2 = LayerNorm(
            d_model=d_model,
            bias=bias,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):

        if use_cache:
            attn_output, present_kv = self.attn(
                self.ln_1(x),
                past_kv=past_kv,
                use_cache=True,
            )

            x = x + attn_output
            x = x + self.ffn(self.ln_2(x))

            return x, present_kv

        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))

        return x