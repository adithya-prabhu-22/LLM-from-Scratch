from pathlib import Path
import shutil
import sys
import torch
from tqdm import tqdm


TOKENIZER_REPO_PATH = Path("/content/Domain-Specific-BPE-Tokenizer")
TOKENIZER_PATH = Path("/content/drive/MyDrive/bpe_medical_52k.json")

TEXT_CHUNKS_DIR = Path("/content/drive/MyDrive/exp_02b_custom_bpe/text_chunks")
TOKENIZED_CHUNKS_DIR = Path("/content/drive/MyDrive/exp_02b_custom_bpe/tokenized_chunks")


def setup_tokenizer_repo() -> None:
    if not TOKENIZER_REPO_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer repo not found: {TOKENIZER_REPO_PATH}. "
            "Clone Domain-Specific-BPE-Tokenizer before running this script."
        )

    sys.path.append(str(TOKENIZER_REPO_PATH))


def main() -> None:
    setup_tokenizer_repo()

    from data.bpe_tokenizer import BPETokenizer

    TOKENIZED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))

    chunk_files = sorted(TEXT_CHUNKS_DIR.glob("*.txt"))

    if not chunk_files:
        raise ValueError(f"No text chunks found in {TEXT_CHUNKS_DIR}")

    total_tokens = 0

    for chunk_file in tqdm(chunk_files, desc="Tokenizing chunks"):
        output_path = TOKENIZED_CHUNKS_DIR / f"{chunk_file.stem}.pt"

        if output_path.exists():
            print(f"Skipping existing: {output_path}")
            continue

        with open(chunk_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)

        tensor = torch.tensor(token_ids, dtype=torch.long)

        torch.save(tensor, output_path)

        total_tokens += len(token_ids)

        print(
            f"Saved {output_path} | "
            f"tokens: {len(token_ids):,} | "
            f"total new tokens: {total_tokens:,}"
        )

    print("\nTokenization complete.")


if __name__ == "__main__":
    main()