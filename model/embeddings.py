import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_len,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:

        B, T = input_ids.shape

        positions = torch.arange(
            start_pos,
            start_pos + T,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        return x