# Experiment 01 — GPT Training Baseline using WikiText-103

## Overview

Experiment 01 is the initial baseline experiment for training a GPT-style decoder-only language model using the GPT-2 BPE tokenizer through `tiktoken` on a subset of the WikiText-103 dataset.

The purpose of this experiment was to validate the complete training workflow before scaling the project to larger datasets, larger model configurations, and more advanced experiment variants.

This experiment was conducted to:

- Validate the end-to-end GPT training pipeline
- Test GPT-2 BPE tokenizer integration using `tiktoken`
- Verify dataset preparation and token counting
- Train a compact decoder-only transformer baseline
- Track training loss, validation loss, and perplexity
- Verify checkpoint saving
- Establish the first baseline for later experiments

---

## Experiment Scope

Experiment 01 was designed as a small-scale baseline. The goal was not to produce a high-quality language model, but to confirm that the full training pipeline works correctly.

The experiment used a compact WikiText-103 subset with approximately 300K tokens. This made it suitable for fast debugging and pipeline validation on limited GPU hardware.

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 01 |
| Dataset | WikiText-103 subset |
| Approximate Token Count | ~300K tokens |
| Tokenizer | GPT-2 BPE tokenizer through `tiktoken` |
| Model Type | GPT-style decoder-only Transformer |
| Objective | Baseline GPT training pipeline validation |
| Hardware | Google Colab T4 GPU |
| Epochs | 5 |
| Checkpoint Strategy | `latest.pt` only |

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 50,257 |
| Context Length | 128 |
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Dropout | 0.1 |
| Approximate Model Size | ~16M parameters |
| Attention Type | Standard causal self-attention |
| Flash Attention | Not used |
| KV Cache Benchmark | Not included |

---

## Hyperparameter Configuration

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning Rate | `3e-4` |
| Minimum Learning Rate | `1e-5` |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Gradient Accumulation | 4 steps |
| Mixed Precision | Enabled |
| Learning Rate Schedule | Cosine decay |
| Epochs | 5 |

---

## Running Experiment 01

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

This creates the training data file:

```text
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
experiments/experiment_01/
```

Expected outputs include:

- Training logs
- Validation logs
- Model checkpoint
- Training plots
- Experiment report

---

## Training Results

| Epoch | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| 1 | 3.4662 | 8.6967 | 5,983 |
| 2 | 0.9319 | 10.7294 | 45,681 |
| 3 | 0.4473 | 11.5839 | 107,358 |
| 4 | 0.3232 | 11.8878 | 145,483 |
| 5 | 0.2834 | 11.9652 | 157,185 |

---

## Result Analysis

Experiment 01 successfully validated the training pipeline, but the model did not generalize well.

The training loss decreased sharply from `3.4662` to `0.2834`, showing that the model learned the training split. However, validation loss increased from `8.6967` to `11.9652`, while perplexity increased from `5,983` to `157,185`.

This indicates severe overfitting. The model memorized the small training subset instead of learning generalizable language patterns.

---

## Failure Diagnosis

| Issue | Severity | Explanation |
|---|---|---|
| Dataset too small | High | ~300K tokens was insufficient for the model size |
| High sliding-window overlap | High | Reduced effective data diversity |
| Context length too small | Medium | 128 tokens limited long-range modeling |
| Checkpointing incomplete | High | Only `latest.pt` was saved |
| No best checkpoint | High | Best validation model could be overwritten |
| No KV-cache benchmark | Medium | Inference efficiency was not evaluated |
| Colab runtime instability | Medium | Long-running reliability was limited |

---

## Key Lessons Learned

Experiment 01 provided important engineering insights:

- Training loss alone is not enough to evaluate model quality.
- Validation loss and perplexity must be monitored carefully.
- Small datasets can cause rapid memorization in transformer models.
- Checkpointing should preserve `best.pt`, `latest.pt`, and epoch checkpoints.
- Larger datasets and longer context lengths are required for better generalization.
- Later experiments should include controlled tokenizer and attention comparisons.

---

## Improvements Required After Experiment 01

| Area | Experiment 01 Limitation | Required Improvement |
|---|---|---|
| Data Scale | ~300K tokens | Larger multi-million-token corpus |
| Domain | General WikiText text | Medical-domain corpus |
| Context Length | 128 | 1024 |
| Checkpointing | `latest.pt` only | `best.pt`, `latest.pt`, and epoch checkpoints |
| Evaluation | Loss and perplexity only | Add KV-cache benchmark |
| Attention | Standard attention only | Add standard vs flash attention comparison |
| Tokenizer | GPT-2 tokenizer only | Compare with custom biomedical BPE |
| Platform | Google Colab T4 | More stable GPU environment |

---

## Transition to Experiment 02A

Experiment 02A was designed as the direct improvement over Experiment 01.

The purpose of Experiment 02A was to keep the tokenizer baseline simple by using the GPT-2 BPE tokenizer through `tiktoken`, while improving the training setup significantly.

| Feature | Experiment 01 | Experiment 02A |
|---|---|---|
| Dataset | WikiText-103 subset | Medical text corpus |
| Token Scale | ~300K tokens | Larger medical corpus |
| Context Length | 128 | 1024 |
| Model Size | ~16M parameters | Larger GPT-style model |
| Checkpointing | `latest.pt` only | `best.pt`, `latest.pt`, epoch checkpoints |
| Logging | Basic logs | Train and validation CSV logs |
| Benchmarking | Not included | KV-cache benchmark |
| Platform | Google Colab T4 | Lightning H200 |

Experiment 02A became the next baseline for the scaled medical-domain GPT training pipeline.

---

## Future Work

The next step after Experiment 01 was to run Experiment 02A, a stronger baseline using a larger GPT-style model, longer context length, improved checkpointing, and a medical-domain dataset.

Experiment 02A was designed to answer the following question:

```text
Can the same GPT training pipeline generalize better when dataset scale,
context length, checkpointing, and training infrastructure are improved?
```

Experiment 02A also prepared the foundation for later comparisons involving custom biomedical tokenization and flash attention.

---

## Conclusion

Experiment 01 successfully validated the core GPT training pipeline, including dataset preparation, GPT-2 BPE tokenization, model training, validation, loss tracking, and checkpoint saving.

However, the experiment showed severe overfitting. Training loss decreased strongly, while validation loss and perplexity increased dramatically. This confirmed that the model memorized the small WikiText-103 subset and failed to generalize.

Although the model quality was poor, Experiment 01 was valuable because it exposed the weaknesses that needed to be fixed before scaling:

- Dataset size
- Context length
- Checkpointing strategy
- Validation behavior
- Training stability
- Experiment tracking

These findings directly motivated Experiment 02A, where the project moved toward a larger medical-domain dataset, stronger checkpointing, longer context length, and more reliable training infrastructure.