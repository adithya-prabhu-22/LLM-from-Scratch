import torch


def create_input_target_pairs(
    tokens,
    context_length: int,
    stride: int = 1,
):

    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
    )

    if len(tokens) <= context_length:
        raise ValueError(
            f"Token sequence length {len(tokens)} must be greater than "
            f"context_length {context_length}"
        )

    if stride <= 0:
        raise ValueError(
            f"stride must be greater than 0, got {stride}"
        )

    input_ids = []
    targets = []

    for i in range(
        0,
        len(tokens) - context_length,
        stride,
    ):
        input_chunk = tokens[i : i + context_length]
        target_chunk = tokens[i + 1 : i + context_length + 1]

        input_ids.append(input_chunk)
        targets.append(target_chunk)

    if not input_ids:
        raise ValueError(
            "No input-target pairs were created. "
            "Use more tokens, smaller context_length, or smaller stride."
        )

    input_ids = torch.stack(input_ids)
    targets = torch.stack(targets)

    return input_ids, targets