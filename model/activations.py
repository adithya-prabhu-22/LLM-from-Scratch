import torch
import torch.nn.functional as F


def swiglu(
    gate: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return F.silu(gate) * value