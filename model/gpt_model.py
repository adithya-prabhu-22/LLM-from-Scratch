import torch
import torch.nn as nn

from model.embeddings import GPTEmbeddings
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.max_len = config.context_length
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.context_length,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    bias=config.qkv_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = LayerNorm(
            d_model=config.d_model,
            bias=config.qkv_bias,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
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