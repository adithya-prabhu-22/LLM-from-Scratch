# Experiment 01 — GPT Training Report

**Dataset:** WikiText-103 (first 5000 rows) — ~300K tokens  
**Hardware:** Google Colab T4  
**Date:** 2026-05-03  
**Epochs:** 5  

---

## Model Configuration

```python
@dataclass
class GPTConfig:
    vocab_size: int                  = 50257
    context_length: int              = 128
    d_model: int                     = 256
    num_heads: int                   = 4
    num_layers: int                  = 4
    dropout: float                   = 0.1
    qkv_bias: bool                   = False
    ffn_hidden_dim: int | None       = None
    max_new_tokens: int              = 100
    temperature: float               = 0.8
    top_k: int | None                = 40
    learning_rate: float             = 3e-4
    min_learning_rate: float         = 1e-5
    grad_clip: float                 = 1.0
    use_amp: bool                    = True
    gradient_accumulation_steps: int = 4
```

**Estimated parameter count: ~16M**

| Component | Params (approx) |
|-----------|----------------|
| Token embedding (`50257 x 256`) | 12.9M |
| Position embedding (`128 x 256`) | 0.03M |
| 4 x Transformer layers (~786K each) | ~3.1M |
| Total | ~16M |

---

## Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Weight decay | 0.1 |
| Flash Attention | Not used |
| AMP (mixed precision) | Enabled |
| Gradient accumulation | 4 steps |
| Gradient clipping | 1.0 |
| LR schedule | Cosine decay `3e-4` to `1e-5` |
| Sliding window overlap | Highly overlapped |
| Checkpoint strategy | `latest.pt` only — overwrites each epoch |

---

## Training Results

| Epoch | Train Loss | Val Loss | Perplexity | LR |
|-------|------------|----------|------------|----|
| 1 | 3.4662 | 8.6967 | 5,983 | 0.000273 |
| 2 | 0.9319 | 10.7294 | 45,681 | 0.000201 |
| 3 | 0.4473 | 11.5839 | 107,358 | 0.000111 |
| 4 | 0.3232 | 11.8878 | 145,483 | 0.000038 |
| 5 | 0.2834 | 11.9652 | 157,185 | 0.000010 |

---

## Diagnosis

Train loss fell consistently across all five epochs while validation loss increased every epoch. The model memorized the training corpus rather than learning generalizable language patterns — a clear case of overfitting. Validation perplexity grew from 5,983 at epoch 1 to 157,185 at epoch 5, confirming that generalization degraded with every additional epoch of training.

The best checkpoint produced by this run was at epoch 1 (val loss 8.6967), which was subsequently overwritten by the `latest.pt` checkpoint strategy.

---

## Root Causes

| Issue | Severity |
|-------|----------|
| 300K tokens substantially below the token scale typically required for strong generalization at this parameter count | High |
| Regularization was insufficient relative to model capacity and dataset size | High |
| Highly overlapped sliding window — reduced effective data diversity | High |
| Dropout set too low at 0.1 for the available data size | Medium |
| Only `latest.pt` saved — best checkpoint at epoch 1 was lost | High |
| Flash Attention not used — missed memory efficiency on T4 | Low |

---

## Recommendations

### 1. Increase dropout

```python
dropout: float = 0.2
```

### 2. Reduce model size or increase data

Option A — shrink the model to match the available data:

```python
d_model: int    = 128
num_layers: int = 2
```

Option B — increase the dataset size. Raising the WikiText-103 slice from `[:5000]` to `[:50000]` rows yields approximately 3M tokens, which is a significantly better fit for the current architecture.

### 3. Reduce sliding window overlap

Experiment with larger strides (for example 32–64) to reduce duplicate subsequences while maintaining sufficient training samples.

### 4. Save per-epoch checkpoints with early stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(checkpoint, "checkpoints/best.pt")

torch.save(checkpoint, f"checkpoints/epoch_{epoch}.pt")
```

Stop training when validation loss has not improved for two consecutive epochs. In this run, training should have stopped after epoch 1.

---

## Proposed Config for Next Run

```python
@dataclass
class GPTConfig:
    vocab_size: int                  = 50257
    context_length: int              = 128
    d_model: int                     = 128       # reduced from 256
    num_heads: int                   = 4
    num_layers: int                  = 2         # reduced from 4
    dropout: float                   = 0.2       # increased from 0.1
    learning_rate: float             = 3e-4
    min_learning_rate: float         = 1e-5
    grad_clip: float                 = 1.0
    use_amp: bool                    = True
    gradient_accumulation_steps: int = 4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=0.1
)
```

---

## Summary

This run confirmed that a ~16M parameter model struggled to generalize effectively on 300K tokens despite the use of weight decay. Insufficient dropout, highly overlapped training windows, and a dataset substantially below the token scale required for this parameter count collectively prevented the model from learning generalizable representations. The best checkpoint from this experiment was produced at epoch 1 and has since been lost due to the single-checkpoint save strategy.

The primary corrective action for the next experiment is acquiring more training data. All other fixes are secondary to data volume.

---

## Future Work

- Integrate and benchmark the custom BPE tokenizer against tiktoken
- Implement Flash Attention and PyTorch SDPA and measure throughput on T4
- Add KV-cache to the inference pipeline and benchmark generation latency
- Scale training to multi-million token regimes using larger WikiText-103 slices or domain-specific corpora
- Evaluate tokenizer efficiency on domain-specific text such as medical or scientific corpora
- Investigate scaling laws empirically by training a grid of model sizes against increasing token budgets
