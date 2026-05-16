from pathlib import Path


INPUT_PATH = Path("resources/data.txt")
OUTPUT_DIR = Path("/content/drive/MyDrive/exp_02b_custom_bpe/text_chunks")

NUM_CHUNKS = 100


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunk_size = len(text) // NUM_CHUNKS

    for i in range(NUM_CHUNKS):
        start = i * chunk_size

        if i == NUM_CHUNKS - 1:
            end = len(text)
        else:
            end = (i + 1) * chunk_size

        chunk_text = text[start:end]

        output_path = OUTPUT_DIR / f"chunk_{i + 1:06d}.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)

        print(f"Saved {output_path} | characters: {len(chunk_text):,}")

    print("\nText chunking complete.")


if __name__ == "__main__":
    main()