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

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Biomedical Corpus

```bash
python scripts/prepare_medical_5m.py
```

This creates:

```text
resources/data.txt
```

### 4. Verify Token Count

```python
import tiktoken

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(text)

print("Total tokens:", len(tokens))
```

Expected output:

```text
~5,000,000 tokens
```

### 5. Train the Model

```bash
python -m training.train
```

### 6. Generate Text

```bash
python -m inference.generate
```

---

## Outputs

Experiment outputs are stored in:

```text
experiments/experiment_02A/
```

Including:

- Training logs
- Validation logs
- Model checkpoints
- Generated text samples
- Training plots
- Experiment report PDF

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 02A |
| Dataset | Biomedical Corpus |
| Dataset Size | ~5M Tokens |
| Tokenizer | GPT-2 TikToken |
| Objective | Medical GPT Training |
| Model Type | Decoder-only Transformer |
| Hardware | NVIDIA T4 GPU |
| Scheduler | Warmup + Cosine Decay |
| Metrics | Validation Loss, Perplexity |

---

## Key Outcome

Experiment 02A establishes the baseline medical-domain language modeling performance using the GPT-2 TikToken tokenizer. The resulting validation loss and perplexity metrics are later compared against Experiment 02B to evaluate the effectiveness of the custom biomedical BPE tokenizer.