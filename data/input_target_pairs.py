import torch


def create_input_target_pairs(
    tokens,
    context_length: int,
):
    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
    )

    input_ids = []
    targets = []

    for i in range(0, len(tokens) - context_length):
        input_chunk = tokens[i : i + context_length]
        target_chunk = tokens[i + 1 : i + context_length + 1]

        input_ids.append(input_chunk)
        targets.append(target_chunk)

    input_ids = torch.stack(input_ids)
    targets = torch.stack(targets)

    return input_ids, targets