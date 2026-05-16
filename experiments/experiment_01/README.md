# Experiment 01 — GPT Training Baseline using WikiText-103

## Overview

Experiment 01 is the initial baseline experiment for training a GPT-style decoder-only language model using the GPT-2 TikToken tokenizer on a small WikiText-103 corpus subset.

This experiment was used to:
- Validate the training pipeline
- Test tokenizer integration
- Verify checkpoint saving
- Establish initial baseline metrics

---

## Run Experiment 01

### 1. Clone Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-from-Scratch.git
cd LLM-from-Scratch
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "wikitext",
    "wikitext-103-raw-v1",
    split="train"
)

text = "\n".join(dataset["text"][:5000])

with open("resources/data.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

This creates:

```
resources/data.txt
```

### 4. Verify Token Count

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
print("Total tokens:", len(tokens))
```

### 5. Train Model

```bash
python -m training.train
```

### 6. Generate Text

```bash
python -m inference.generate
```

---

## Outputs

Experiment outputs are saved in:

```
experiments/exp_01_baseline/
```

Including:
- Train logs
- Validation logs
- Checkpoints
- Generated samples

---

## Experiment Summary

| Component  | Value                    |
| ---------- | ------------------------ |
| Dataset    | WikiText-103 subset      |
| Tokenizer  | GPT-2 TikToken           |
| Objective  | Baseline GPT training    |
| Model Type | Decoder-only Transformer |
| Hardware   | Google Colab T4          |