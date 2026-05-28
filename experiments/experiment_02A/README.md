# Experiment 02A — Medical GPT Training Baseline using GPT-2/tiktoken

## Overview

Experiment 02A is the scaled medical-domain baseline experiment for the GPT-style decoder-only language model. It uses the GPT-2 BPE tokenizer through `tiktoken` and standard causal multi-head self-attention.

This experiment was designed as the direct improvement over Experiment 01. The goal was to move from a small WikiText-103 debugging baseline to a larger medical-domain training setup with improved checkpointing, longer context length, stronger logging, and KV-cache inference benchmarking.

Experiment 02A establishes the main GPT-2/tiktoken baseline for medical-domain language modeling. A later experiment extends this baseline by replacing the generic tokenizer with a custom biomedical BPE tokenizer while keeping the training setup comparable.

This experiment was conducted to:

- Train a larger GPT-style decoder-only Transformer on medical-domain text
- Establish a GPT-2/tiktoken baseline for medical language modeling
- Measure training loss, validation loss, and perplexity across epochs
- Verify stable checkpoint preservation using `best.pt`, `latest.pt`, and epoch checkpoints
- Benchmark autoregressive inference with and without KV-cache
- Prepare a controlled baseline for later tokenizer comparison

---

## Experiment Scope

Experiment 02A focuses only on the GPT-2/tiktoken tokenizer baseline.

It does not use a custom biomedical tokenizer. That comparison is reserved for the next experiment.

| Experiment | Tokenizer | Attention | Purpose |
|---|---|---|---|
| Experiment 02A | GPT-2 BPE through `tiktoken` | Standard causal attention | Medical-domain baseline |
| Experiment 02B | Custom biomedical BPE | Standard causal attention | Tokenizer comparison |

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 02A |
| Dataset | Medical text corpus |
| Dataset Format | Text chunks |
| Tokenizer | GPT-2 BPE tokenizer through `tiktoken` |
| Vocabulary Size | 50,257 |
| Model Type | GPT-style decoder-only Transformer |
| Attention Type | Standard causal multi-head self-attention |
| Hardware | Lightning AI H200 |
| Epochs | 5 |
| Batch Size | 8 |
| Gradient Accumulation | 8 steps |
| Effective Batch Size | 64 |
| Context Length | 1024 |
| Objective | Medical-domain GPT baseline training |
| Benchmark | KV-cache inference benchmark |

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 50,257 |
| Context Length | 1024 |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Feedforward Hidden Dimension | 3072 |
| Dropout | 0.1 |
| QKV Bias | False |
| Approximate Model Size | ~194M parameters |
| Attention Type | Standard causal self-attention |
| KV Cache | Benchmarked after training |

---

## Hyperparameter Configuration

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning Rate | `3e-4` |
| Minimum Learning Rate | `1e-5` |
| Scheduler | Warmup + cosine decay |
| Gradient Clipping | 1.0 |
| Gradient Accumulation | 8 steps |
| Batch Size | 8 |
| Effective Batch Size | 64 |
| Mixed Precision | Enabled |
| Epochs | 5 |
| Checkpoint Strategy | `best.pt`, `latest.pt`, and `epoch_N.pt` |

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
pip install tiktoken
```

### 3. Prepare Medical Text Chunks

Experiment 02A expects medical-domain text chunks as input.

The final training run used text chunks copied into the following structure:

```text
exp_02b_custom_bpe/
└── text_chunks/
```

Example path used during final training:

```text
/teamspace/studios/this_studio/exp_02b_custom_bpe/text_chunks
```

### 4. Train the Model

Run Experiment 02A using the dedicated training entry point:

```bash
python -m training.train_experiment_02a \
  --data_dir /teamspace/studios/this_studio/exp_02b_custom_bpe/text_chunks \
  --num_epochs 5 \
  --batch_size 8
```

For a local setup, replace the `--data_dir` path with the location of the prepared medical text chunks.

---

## Outputs

Experiment outputs are stored in:

```text
experiments/exp_02a_tiktoken_medical_5m/
```

Expected outputs include:

- `config.json`
- `train_log.csv`
- `val_log.csv`
- `kv_cache_benchmark.csv`
- `checkpoints/best.pt`
- `checkpoints/latest.pt`
- `checkpoints/epoch_N.pt`

For GitHub documentation, only the report, README, plots, and lightweight metadata should be committed. Large checkpoint files should be excluded from Git.

---

## Training Results

| Epoch | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| 1 | 6.4186 | 5.1973 | 180.7757 |
| 2 | 4.5894 | 4.6690 | 106.5914 |
| 3 | 3.9408 | 4.3225 | 75.3733 |
| 4 | 3.5148 | 4.2077 | 67.2047 |
| 5 | 3.3203 | 4.1917 | 66.1369 |

---

## Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 3.3203 |
| Final Validation Loss | 4.1917 |
| Final Perplexity | 66.1369 |
| Best Checkpoint | Epoch 5 |
| Best Checkpoint File | `best.pt` |

Experiment 02A showed stable training behavior. Validation loss decreased from `5.1973` to `4.1917`, and perplexity decreased from `180.7757` to `66.1369`.

This confirms that the larger medical-domain setup generalized significantly better than the small Experiment 01 baseline.

---

## KV-Cache Benchmark

After training, Experiment 02A was evaluated using an autoregressive inference benchmark with and without KV-cache.

### Benchmark Command

```bash
python -m inference.benchmark_kv_cache \
  --config experiments/exp_02a_tiktoken_medical_5m/config.json \
  --checkpoint experiments/exp_02a_tiktoken_medical_5m/checkpoints/best.pt \
  --output experiments/exp_02a_tiktoken_medical_5m/kv_cache_benchmark.csv \
  --prompt_lengths 64,128,256,512 \
  --max_new_tokens 64
```

### KV-Cache Results

| Prompt Length | No KV Latency | KV Latency | Speedup | No KV Tokens/sec | KV Tokens/sec |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.333337 | 0.303964 | 1.0966x | 192.00 | 210.56 |
| 128 | 0.334250 | 0.306064 | 1.0921x | 191.50 | 209.13 |
| 256 | 0.328982 | 0.302917 | 1.0860x | 194.55 | 211.29 |
| 512 | 0.456390 | 0.296575 | 1.5389x | 140.23 | 215.80 |

---

## KV-Cache Analysis

KV-cache improved inference throughput most clearly at longer prompt lengths.

At prompt length `512`, latency decreased from `0.456390s` to `0.296575s`, producing a `1.5389x` speedup. Throughput increased from `140.23` tokens/sec to `215.80` tokens/sec.

This confirms that KV-cache becomes more valuable as the prompt length increases because the model avoids recomputing attention over previously processed tokens during autoregressive generation.

---

## Plots

Training and evaluation plots are available in the `plots/` directory.

Recommended plots for this experiment include:

- Training vs validation loss
- Validation perplexity
- Learning-rate schedule
- Generalization gap
- KV-cache speedup
- KV-cache throughput comparison
- KV-cache latency comparison
- KV-cache memory comparison

---

## Result Analysis

Experiment 02A successfully corrected the major issues observed in Experiment 01.

Compared with Experiment 01, this experiment used:

- A larger medical-domain corpus
- A longer context length
- A larger GPT-style model configuration
- Better checkpoint preservation
- CSV-based training and validation logs
- KV-cache inference benchmarking
- A more stable Lightning H200 training environment

The validation loss decreased consistently across epochs, showing that the model did not suffer the same severe overfitting pattern observed in Experiment 01.

However, because Experiment 02A uses the generic GPT-2/tiktoken tokenizer, it remains a baseline. Medical-domain terms may still be split inefficiently compared with a domain-specific tokenizer.

---

## Future Work

The next experiment extends this baseline by changing the tokenizer while keeping the training setup comparable.

The main planned change is to replace the generic GPT-2/tiktoken tokenizer with a custom biomedical BPE tokenizer. This allows the project to evaluate whether domain-specific tokenization improves medical language modeling performance on the same corpus.

The next experiment focuses on the following question:

```text
Can a biomedical-domain tokenizer improve validation loss and perplexity
compared with the generic GPT-2/tiktoken tokenizer baseline?
```

---

## Conclusion

Experiment 02A established a strong scaled baseline for medical-domain GPT training using the GPT-2 BPE tokenizer through `tiktoken`.

The experiment achieved stable validation behavior across five epochs, with validation loss decreasing from `5.1973` to `4.1917` and perplexity decreasing from `180.7757` to `66.1369`.

The KV-cache benchmark confirmed that cached autoregressive decoding improves inference efficiency, especially at longer prompt lengths. The best observed KV-cache speedup was `1.5389x` at prompt length `512`.

Overall, Experiment 02A provides the main GPT-2/tiktoken medical-domain baseline for the next tokenizer-focused experiment.