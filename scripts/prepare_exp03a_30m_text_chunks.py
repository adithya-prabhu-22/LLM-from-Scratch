from pathlib import Path
import shutil


SOURCE_ROOT = Path("/content/drive/MyDrive/final_corpus")
EXP3A_ROOT = Path("/content/drive/MyDrive/exp3a")

PMC_DIR = SOURCE_ROOT / "pmc_open"
GENERAL_DIR = SOURCE_ROOT / "general"
PUBMED_DIR = SOURCE_ROOT / "pubmed"

RAW_1M_DIR = EXP3A_ROOT / "raw_1m_chunks"
RAW_50K_DIR = EXP3A_ROOT / "raw_50k_chunks"

PMC_CHUNKS = 24
GENERAL_CHUNKS = 4
PUBMED_CHUNKS = 2

SUBCHUNKS_PER_FILE = 20


def get_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file()])


def copy_files(files: list[Path], count: int, prefix: str) -> list[Path]:
    selected = files[:count]

    if len(selected) < count:
        raise ValueError(f"Not enough {prefix} files.")

    copied = []

    for idx, src in enumerate(selected, start=1):
        dst = RAW_1M_DIR / f"{prefix}_{idx:06d}_{src.name}"

        if not dst.exists():
            shutil.copy2(src, dst)

        copied.append(dst)

    return copied


def split_file(input_file: Path, start_idx: int) -> int:
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunk_size = max(1, len(text) // SUBCHUNKS_PER_FILE)
    current_idx = start_idx

    for i in range(SUBCHUNKS_PER_FILE):
        start = i * chunk_size
        end = len(text) if i == SUBCHUNKS_PER_FILE - 1 else (i + 1) * chunk_size

        chunk_text = text[start:end]
        output_path = RAW_50K_DIR / f"chunk_{current_idx:06d}.txt"

        if not output_path.exists():
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(chunk_text)

        print(f"Saved {output_path.name} | chars: {len(chunk_text):,}")

        current_idx += 1

    return current_idx


def main() -> None:
    RAW_1M_DIR.mkdir(parents=True, exist_ok=True)
    RAW_50K_DIR.mkdir(parents=True, exist_ok=True)

    pmc_files = get_files(PMC_DIR)
    general_files = get_files(GENERAL_DIR)
    pubmed_files = get_files(PUBMED_DIR)

    selected_files = []
    selected_files.extend(copy_files(pmc_files, PMC_CHUNKS, "pmc"))
    selected_files.extend(copy_files(general_files, GENERAL_CHUNKS, "general"))
    selected_files.extend(copy_files(pubmed_files, PUBMED_CHUNKS, "pubmed"))

    print(f"Selected raw chunks: {len(selected_files)}")

    chunk_idx = 1

    for file in selected_files:
        print(f"\nSplitting {file.name}")
        chunk_idx = split_file(file, chunk_idx)

    print("\nExp03A text chunk preparation complete.")
    print(f"Raw 1M chunks: {RAW_1M_DIR}")
    print(f"Raw 50K chunks: {RAW_50K_DIR}")
    print(f"Total 50K chunks: {chunk_idx - 1}")


if __name__ == "__main__":
    main()