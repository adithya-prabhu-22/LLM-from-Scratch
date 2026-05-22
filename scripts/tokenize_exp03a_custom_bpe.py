from pathlib import Path
import sys
import json
import torch
from tqdm import tqdm


TOKENIZER_REPO_PATH = Path("/content/Domain-Specific-BPE-Tokenizer")
TOKENIZER_PATH = Path("/content/drive/MyDrive/bpe_medical_52k.json")

EXP3A_ROOT = Path("/content/drive/MyDrive/exp3a")
TEXT_CHUNKS_DIR = EXP3A_ROOT / "raw_50k_chunks"
TOKENIZED_CHUNKS_DIR = EXP3A_ROOT / "tokenized_50k_chunks"
METADATA_DIR = EXP3A_ROOT / "metadata"


def setup_tokenizer_repo() -> None:
    if not TOKENIZER_REPO_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer repository not found: {TOKENIZER_REPO_PATH}"
        )

    if str(TOKENIZER_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(TOKENIZER_REPO_PATH))


def main() -> None:
    setup_tokenizer_repo()

    from data.bpe_tokenizer import BPETokenizer

    TOKENIZED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))

    chunk_files = sorted(TEXT_CHUNKS_DIR.glob("*.txt"))

    if not chunk_files:
        raise ValueError(f"No text chunks found in {TEXT_CHUNKS_DIR}")

    total_new_tokens = 0
    processed_files = 0

    for chunk_file in tqdm(chunk_files, desc="Tokenizing Exp03A chunks"):
        output_path = TOKENIZED_CHUNKS_DIR / f"{chunk_file.stem}.pt"
        metadata_path = METADATA_DIR / f"{chunk_file.stem}.json"

        if output_path.exists():
            print(f"Skipping existing: {output_path.name}")
            continue

        with open(chunk_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)
        tensor = torch.tensor(token_ids, dtype=torch.long)

        torch.save(tensor, output_path)

        meta = {
            "source_file": chunk_file.name,
            "output_file": output_path.name,
            "token_count": len(token_ids),
            "tokenizer": str(TOKENIZER_PATH),
            "status": "done",
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        total_new_tokens += len(token_ids)
        processed_files += 1

        print(
            f"Saved {output_path.name} | "
            f"tokens: {len(token_ids):,} | "
            f"total new tokens: {total_new_tokens:,}"
        )

    manifest = {
        "text_chunks_dir": str(TEXT_CHUNKS_DIR),
        "tokenized_chunks_dir": str(TOKENIZED_CHUNKS_DIR),
        "tokenizer_path": str(TOKENIZER_PATH),
        "new_files_processed": processed_files,
        "total_new_tokens": total_new_tokens,
    }

    with open(METADATA_DIR / "tokenization_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nExp03A tokenization complete.")
    print(f"New files processed: {processed_files}")
    print(f"New tokens: {total_new_tokens:,}")
    print(f"Output folder: {TOKENIZED_CHUNKS_DIR}")


if __name__ == "__main__":
    main()