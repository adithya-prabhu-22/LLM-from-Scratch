import torch
from torch.utils.data import Dataset, DataLoader

from data.input_target_pairs import create_input_target_pairs


class GPTDataset(Dataset):

    def __init__(
        self,
        tokens,
        context_length: int,
        stride: int = 1,
    ):
        self.input_ids, self.targets = create_input_target_pairs(
            tokens=tokens,
            context_length=context_length,
            stride=stride,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(
        self,
        idx,
    ):
        return self.input_ids[idx], self.targets[idx]


def create_dataloader(
    tokens,
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    stride: int = 1,
):

    dataset = GPTDataset(
        tokens=tokens,
        context_length=context_length,
        stride=stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader