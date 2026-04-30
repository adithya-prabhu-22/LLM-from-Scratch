import torch
import torch.nn as nn

from model.embeddings import GPTEmbeddings
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(
            d_model=d_model,
            bias=bias,
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len {self.max_len}"
            )

        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits