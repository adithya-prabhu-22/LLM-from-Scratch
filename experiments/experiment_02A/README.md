Use **no `id="..."` inside code fences**. GitHub renders those badly sometimes.

Start Exp02A README like this:

````markdown
# Experiment 02A — Medical GPT Training using GPT-2 TikToken

## Overview

Experiment 02A evaluates a GPT-style decoder-only Transformer trained on a ~5 million token biomedical corpus using the GPT-2 TikToken tokenizer.

This experiment establishes the tokenizer and training baseline for comparison against Experiment 02B, which replaces TikToken with a custom biomedical BPE tokenizer while keeping the training configuration unchanged.

This experiment was conducted to:

- Evaluate medical-domain language modeling performance
- Measure validation convergence behavior
- Establish baseline perplexity metrics
- Analyze training stability on biomedical text
- Benchmark GPT-2 TikToken against the custom tokenizer

---

## Running Experiment 02A

### 1. Clone Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-from-Scratch.git
cd LLM-from-Scratch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Biomedical Corpus

```bash
python scripts/prepare_medical_5m.py
```

This generates:

```text
resources/data.txt
```

### 4. Verify Dataset Token Count

```python
import tiktoken

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(text)

print(f"Total Tokens: {len(tokens):,}")
```

Expected output:

```text
~5,000,000 tokens
```

### 5. Start Training

```bash
python -m training.train
```

### 6. Generate Text Samples

```bash
python -m inference.generate
```

---

## Experiment Outputs

All experiment artifacts are stored in:

```text
experiments/experiment_02A/
```

Generated outputs include:

```text
plots/
README.md
Experiment_02A_report.pdf
checkpoints/
train_log.csv
val_log.csv
tokenizer_report.json
```

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 02A |
| Dataset Domain | Biomedical / Medical |
| Dataset Size | ~5M Tokens |
| Tokenizer | GPT-2 TikToken |
| Model Type | Decoder-only Transformer |
| Training Objective | Medical Language Modeling |
| Training Hardware | NVIDIA T4 GPU |
| Scheduler | Warmup + Cosine Decay |
| Metrics | Validation Loss, Perplexity |
| Baseline Purpose | Comparison against Custom BPE Tokenizer |

---

## Key Outcome

Experiment 02A establishes the baseline medical-domain language modeling performance using the GPT-2 TikToken tokenizer. The results from this experiment are directly compared against Experiment 02B to evaluate the effectiveness of the custom biomedical BPE tokenizer.
````
