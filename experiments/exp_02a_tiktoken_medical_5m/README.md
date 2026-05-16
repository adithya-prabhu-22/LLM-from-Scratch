# Experiment 02A — TikToken Medical LLM Baseline

## Overview

Experiment 02A trains a GPT-style decoder-only language model on a ~5M-token medical corpus using the GPT-2 TikToken tokenizer.

This experiment is the baseline for Experiment 02B, where the same model and dataset will be trained using a custom biomedical BPE tokenizer.

---

## Run Experiment 02A

### 1. Clone Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-from-Scratch.git
cd LLM-from-Scratch
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare Medical Dataset

```bash
python scripts/prepare_medical_5m.py
```

This creates:

```
resources/data.txt
```

### 4. Verify Token Count

```python
import tiktoken

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
print(f"Tokens: {len(enc.encode(text)):,}")
```

Expected output: approximately 5M tokens.

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

Experiment files are saved in:

```
experiments/exp_02a_tiktoken_medical_5m/
```

Main outputs:

```
train_log.csv
val_log.csv
tokenizer_report.json
checkpoints/best.pt
```

---

## Colab Download

```python
from google.colab import files

files.download("experiments/exp_02a_tiktoken_medical_5m/train_log.csv")
files.download("experiments/exp_02a_tiktoken_medical_5m/val_log.csv")
files.download("experiments/exp_02a_tiktoken_medical_5m/tokenizer_report.json")
files.download("experiments/exp_02a_tiktoken_medical_5m/checkpoints/best.pt")
```