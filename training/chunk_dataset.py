from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenChunkDataset(Dataset):

    def __init__(
        self,
        tokenized_dir: str,
        context_length: int,
        stride: int,
        max_cached_chunks: int = 8,
    ):
        self.tokenized_dir = Path(tokenized_dir)
        self.context_length = context_length
        self.stride = stride
        self.max_cached_chunks = max_cached_chunks

        self.samples = []
        self.cache = {}

        chunk_files = sorted(self.tokenized_dir.glob("*.pt"))

        if not chunk_files:
            raise ValueError(
                f"No tokenized chunks found in {self.tokenized_dir}"
            )

        for chunk_path in chunk_files:
            tokens = torch.load(chunk_path, map_location="cpu")
            num_tokens = len(tokens)

            for start in range(0, num_tokens - context_length, stride):
                self.samples.append((chunk_path, start))

        print(f"Loaded tokenized chunks from: {self.tokenized_dir}")
        print(f"Chunk files: {len(chunk_files)}")
        print(f"Total samples: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def _load_chunk(self, chunk_path: Path):
        if chunk_path in self.cache:
            return self.cache[chunk_path]

        tokens = torch.load(chunk_path, map_location="cpu")

        if len(self.cache) >= self.max_cached_chunks:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[chunk_path] = tokens

        return tokens

    def __getitem__(self, index):
        chunk_path, start = self.samples[index]

        tokens = self._load_chunk(chunk_path)

        x = tokens[start : start + self.context_length]
        y = tokens[start + 1 : start + self.context_length + 1]

        return x, y