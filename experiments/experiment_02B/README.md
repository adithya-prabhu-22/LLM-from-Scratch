# Experiment 02B — Medical GPT Training using Custom Biomedical BPE Tokenizer

## Overview

Experiment 02B evaluates a GPT-style decoder-only Transformer trained on a ~5 million token biomedical corpus using a custom biomedical BPE tokenizer.

This experiment serves as a controlled comparison against Experiment 02A. The dataset, model architecture, training configuration, optimizer, scheduler, and evaluation methodology remain unchanged, with the tokenizer being the only modified component.

This experiment was conducted to:

- Evaluate the effectiveness of domain-specific tokenization
- Measure convergence behavior using a custom tokenizer
- Compare validation loss and perplexity against GPT-2 TikToken
- Analyze training stability on biomedical text
- Quantify the impact of tokenizer design on language modeling performance

---

## Running Experiment 02B

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

### 4. Tokenize using Custom Biomedical BPE

```bash
python scripts/tokenize_chunks_custom_bpe.py
```

This generates tokenized biomedical training data using the custom BPE tokenizer trained on a large biomedical corpus.

### 5. Train the Model

```bash
python -m training.train_experiment_02b
```

### 6. Generate Text

```bash
python -m inference.generate
```

---

## Outputs

Experiment outputs are stored in:

```text
experiments/experiment_02B/
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
| Experiment | Experiment 02B |
| Dataset | Biomedical Corpus |
| Dataset Size | ~5M Tokens |
| Tokenizer | Custom Biomedical BPE |
| Vocabulary Size | ~52,000 Tokens |
| Objective | Medical GPT Training |
| Model Type | Decoder-only Transformer |
| Hardware | NVIDIA T4 GPU |
| Scheduler | Warmup + Cosine Decay |
| Metrics | Validation Loss, Perplexity |

---

## Key Outcome

Experiment 02B demonstrates the effectiveness of domain-specific tokenization for biomedical language modeling. Under identical training conditions, the custom biomedical BPE tokenizer achieved significantly lower validation loss and perplexity than the GPT-2 TikToken baseline from Experiment 02A, indicating more efficient representation of medical terminology and improved language modeling performance.

The final validation loss of **2.0155** and perplexity of **7.5042** establish the strongest results obtained within the current experimental series and validate the effectiveness of the custom tokenizer design.