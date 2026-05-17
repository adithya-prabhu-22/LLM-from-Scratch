````markdown
# Experiment 02A — TikToken Medical LLM Baseline

## Overview

Experiment 02A evaluates a GPT-style decoder-only Transformer trained on a ~5 million token biomedical corpus using the GPT-2 TikToken tokenizer.

This experiment establishes the tokenizer and training baseline for comparison against Experiment 02B, which replaces TikToken with a custom domain-specific biomedical BPE tokenizer while keeping the training configuration unchanged.

The primary objective of this experiment is to evaluate:
- Training stability
- Validation convergence
- Language modeling performance
- Baseline perplexity metrics
- Tokenization efficiency using GPT-2 TikToken

---

# Repository Setup

## 1. Clone Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-from-Scratch.git
cd LLM-from-Scratch
````

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

## 3. Prepare Biomedical Corpus

```bash
python scripts/prepare_medical_5m.py
```

This generates:

```text
resources/data.txt
```

The dataset consists of biomedical and medical-domain text collected primarily from PubMed abstracts.

---

# Token Verification

## 4. Verify Dataset Token Count

```python
import tiktoken

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

tokens = tokenizer.encode(text)

print(f"Total Tokens: {len(tokens):,}")
```

Expected token count:

```text
~5,000,000 tokens
```

---

# Model Training

## 5. Start Training

```bash
python -m training.train
```

Training includes:

* Decoder-only Transformer architecture
* Causal masked self-attention
* Mixed precision training
* Gradient clipping
* Warmup + cosine learning rate scheduling
* Validation perplexity evaluation
* Automatic checkpoint saving

---

# Text Generation

## 6. Generate Text Samples

```bash
python -m inference.generate
```

This loads the trained checkpoint and performs autoregressive text generation.

---

# Experiment Outputs

All experiment artifacts are stored in:

```text
experiments/Experiment_02A/
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

# Colab Download Commands

```python
from google.colab import files

files.download("experiments/Experiment_02A/train_log.csv")
files.download("experiments/Experiment_02A/val_log.csv")
files.download("experiments/Experiment_02A/tokenizer_report.json")
files.download("experiments/Experiment_02A/checkpoints/best.pt")
```

---

# Experiment Summary

| Component          | Value                                   |
| ------------------ | --------------------------------------- |
| Experiment         | Experiment 02A                          |
| Dataset Domain     | Biomedical / Medical                    |
| Dataset Size       | ~5M Tokens                              |
| Tokenizer          | GPT-2 TikToken                          |
| Model Type         | Decoder-only Transformer                |
| Training Objective | Medical Language Modeling               |
| Training Hardware  | NVIDIA T4 GPU (Google Colab)            |
| Scheduler          | Warmup + Cosine Decay                   |
| Evaluation Metrics | Validation Loss, Perplexity             |
| Baseline Purpose   | Comparison against Custom BPE Tokenizer |

---

# Key Outcome

Experiment 02A establishes the baseline medical-domain language modeling performance using the GPT-2 TikToken tokenizer. The results from this experiment are directly compared against Experiment 02B to evaluate the effectiveness of the custom biomedical BPE tokenizer.

```
```
