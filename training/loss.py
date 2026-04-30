import torch
import torch.nn as nn


class GPTLoss(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        B, T, V = logits.shape

        logits = logits.view(B * T, V)

        targets = targets.view(B * T)

        loss = self.loss_fn(logits, targets)

        return loss