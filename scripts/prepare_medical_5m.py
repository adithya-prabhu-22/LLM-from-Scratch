from datasets import load_dataset
from pathlib import Path
import tiktoken


OUTPUT_PATH = "resources/data.txt"
TARGET_TOKENS = 5_000_000
TOKENIZER_NAME = "gpt2"


def main():
    Path("resources").mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)

    dataset = load_dataset(
        "ncbi/pubmed",
        split="train",
        streaming=True,
    )

    total_tokens = 0
    total_docs = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row.get("abstract", "")

            if text is None:
                continue

            text = text.strip()

            if len(text) < 200:
                continue

            num_tokens = len(tokenizer.encode(text))

            f.write(text.replace("\n", " ") + "\n\n")

            total_tokens += num_tokens
            total_docs += 1

            if total_docs % 1000 == 0:
                print(
                    f"Docs: {total_docs:,} | "
                    f"Tokens: {total_tokens:,}"
                )

            if total_tokens >= TARGET_TOKENS:
                break

    print(f"Saved to {OUTPUT_PATH}")
    print(f"Total docs: {total_docs:,}")
    print(f"Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()