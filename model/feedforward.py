import torch
import torch.nn as nn

from model.activations import swiglu


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = swiglu(
            self.gate_proj(x),
            self.up_proj(x),
        )

        x = self.down_proj(x)
        x = self.dropout(x)

        return x