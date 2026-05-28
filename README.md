# LLM-from-Scratch

A from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch.

This project implements the core components of a modern autoregressive language model without relying on external Transformer libraries. It includes tokenizer integration, causal multi-head self-attention, Flash/SDPA attention support, SwiGLU feedforward blocks, training loops, checkpointing, validation, sampling, and KV-cache inference benchmarking.

The project evolved from a small WikiText baseline into a controlled medical-domain GPT experiment series comparing tokenizer quality, attention implementation, and inference efficiency.

---

## Project Overview

This repository focuses on building and evaluating a GPT-style language model from scratch.

Core goals:

- Implement a decoder-only Transformer architecture in PyTorch
- Train GPT-style language models on text corpora
- Compare tokenizer choices for domain-specific language modeling
- Evaluate standard attention and Flash/SDPA attention
- Add checkpointing, logging, validation, and reproducible experiment folders
- Benchmark autoregressive inference with and without KV-cache
- Document each experiment with plots, reports, and README files

---

## Key Features

- GPT-style decoder-only Transformer
- Token embedding and positional embedding layers
- Causal multi-head self-attention
- Standard attention and Flash/SDPA attention support
- SwiGLU feedforward network
- Pre-layer normalization Transformer blocks
- Autoregressive next-token prediction objective
- Mixed precision training
- Gradient accumulation
- Gradient clipping
- Warmup + cosine learning-rate scheduling
- Training and validation logging
- Best, latest, and epoch-wise checkpoint saving
- KV-cache inference benchmark
- Experiment-specific reports, plots, and README files

---

## Repository Structure

```text
LLM-from-Scratch/
├── config/
│   └── config.py
│
├── data/
│   └── dataset utilities
│
├── inference/
│   ├── generate.py
│   └── benchmark_kv_cache.py
│
├── model/
│   ├── attention/
│   │   └── multihead_attention.py
│   ├── feedforward.py
│   ├── gpt_model.py
│   └── transformer_block.py
│
├── scripts/
│   └── data preparation scripts
│
├── training/
│   ├── train.py
│   ├── train_experiment_02a.py
│   ├── train_experiment_02b.py
│   └── train_experiment_02c.py
│
├── experiments/
│   ├── experiment_01/
│   │   ├── README.md
│   │   ├── Experiment_01_report.pdf
│   │   └── plots/
│   │
│   ├── experiment_02A/
│   │   ├── README.md
│   │   ├── Experiment_02A_report.pdf
│   │   └── plots/
│   │
│   ├── experiment_02B/
│   │   ├── README.md
│   │   ├── Experiment_02B_report.pdf
│   │   └── plots/
│   │
│   └── experiment_02C/
│       ├── README.md
│       ├── Experiment_02C_report.pdf
│       └── plots/
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The `experiment_01`, `experiment_02A`, `experiment_02B`, and `experiment_02C` folders are documentation folders. Runtime training outputs use separate generated folders such as `exp_02a_tiktoken_medical_5m`, `exp_02b_custom_bpe_medical_5m`, and `exp_02c_custom_bpe_flash_attention_medical_5m`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adithya-prabhu-22/LLM-from-Scratch.git
cd LLM-from-Scratch
```

### 2. Install Core Requirements

```bash
pip install -r requirements.txt
```

### 3. Install Development Requirements

```bash
pip install -r requirements-dev.txt
```

---

## Requirements

### `requirements.txt`

```text
torch
numpy
tiktoken
regex
requests
tqdm
```

### `requirements-dev.txt`

```text
-r requirements.txt

matplotlib
networkx
sympy
jupyter
pytest
tensorboard
```

`rclone` is not included in the Python requirements because it is a system-level command-line tool, not a Python package. It is only used optionally for cloud-to-Google-Drive backup workflows.

---

## Experiments

This repository contains four documented experiments.

| Experiment | Description | Tokenizer | Attention | Purpose |
|---|---|---|---|---|
| Experiment 01 | WikiText-103 baseline | GPT-2 BPE / `tiktoken` | Standard | Initial pipeline validation |
| Experiment 02A | Medical GPT baseline | GPT-2 BPE / `tiktoken` | Standard | Scaled medical-domain baseline |
| Experiment 02B | Custom tokenizer experiment | Custom biomedical BPE | Standard | Tokenizer quality comparison |
| Experiment 02C | Flash attention experiment | Custom biomedical BPE | Flash/SDPA | Attention efficiency comparison |

---

## Experiment 01 — WikiText-103 Baseline

Experiment 01 was the initial baseline. It used a small WikiText-103 subset with approximately 300K tokens and a compact GPT-style model.

The goal was to validate the full training pipeline:

- Dataset loading
- GPT-2 BPE tokenization
- Sliding-window dataset creation
- Decoder-only Transformer training
- Validation loop
- Checkpoint saving
- Loss and perplexity tracking

### Experiment 01 Result

| Epoch | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| 1 | 3.4662 | 8.6967 | 5,983 |
| 2 | 0.9319 | 10.7294 | 45,681 |
| 3 | 0.4473 | 11.5839 | 107,358 |
| 4 | 0.3232 | 11.8878 | 145,483 |
| 5 | 0.2834 | 11.9652 | 157,185 |

Experiment 01 successfully validated the pipeline but showed severe overfitting. Training loss decreased sharply while validation loss and perplexity increased, showing that the model memorized the small dataset instead of generalizing.

---

## Experiment 02A — Medical GPT Baseline using GPT-2/tiktoken

Experiment 02A scaled the project from the small WikiText baseline to a larger medical-domain training setup.

It used:

- Medical text corpus
- GPT-2 BPE tokenizer through `tiktoken`
- Standard causal multi-head attention
- Larger GPT-style model configuration
- Context length of 1024
- Better checkpointing
- KV-cache inference benchmark
- Lightning AI H200 training environment

### Experiment 02A Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 3.3203 |
| Final Validation Loss | 4.1917 |
| Final Perplexity | 66.1369 |
| Best Checkpoint | Epoch 5 |
| Best KV-cache Speedup | 1.5389x |

### Experiment 02A Epoch Results

| Epoch | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| 1 | 6.4186 | 5.1973 | 180.7757 |
| 2 | 4.5894 | 4.6690 | 106.5914 |
| 3 | 3.9408 | 4.3225 | 75.3733 |
| 4 | 3.5148 | 4.2077 | 67.2047 |
| 5 | 3.3203 | 4.1917 | 66.1369 |

Experiment 02A became the main GPT-2/tiktoken baseline for medical-domain language modeling.

---

## Experiment 02B — Custom Biomedical BPE Tokenizer

Experiment 02B replaced the GPT-2/tiktoken tokenizer with a custom biomedical BPE tokenizer while keeping the training setup comparable to Experiment 02A.

The goal was to evaluate whether domain-specific tokenization improves medical-domain language modeling performance.

### Experiment 02B Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 2.0879 |
| Final Validation Loss | 2.1301 |
| Final Perplexity | 8.4161 |
| Best Checkpoint | Epoch 5 |
| Best KV-cache Speedup | 1.4896x |

### Experiment 02B Epoch Results

| Epoch | Optimizer Step | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|---:|
| 1 | 420 | 4.1228 | 3.5888 | 36.1901 |
| 2 | 840 | 3.0433 | 2.6425 | 14.0488 |
| 3 | 1260 | 2.4642 | 2.3138 | 10.1131 |
| 4 | 1680 | 2.2049 | 2.1706 | 8.7634 |
| 5 | 2100 | 2.0879 | 2.1301 | 8.4161 |

### Tokenizer Comparison

| Metric | Experiment 02A: GPT-2/tiktoken | Experiment 02B: Custom Biomedical BPE |
|---|---:|---:|
| Final Train Loss | 3.3203 | 2.0879 |
| Final Validation Loss | 4.1917 | 2.1301 |
| Final Perplexity | 66.1369 | 8.4161 |

Experiment 02B significantly improved validation loss and perplexity. This shows that the custom biomedical BPE tokenizer was much better aligned with the medical-domain corpus than the generic GPT-2 tokenizer.

---

## Experiment 02C — Custom Biomedical BPE with Flash/SDPA Attention

Experiment 02C kept the custom biomedical BPE tokenizer from Experiment 02B but replaced the standard attention path with Flash/SDPA attention.

The goal was to evaluate whether an optimized attention implementation could preserve the language modeling quality of Experiment 02B while improving systems-level efficiency.

### Experiment 02C Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 2.1006 |
| Final Validation Loss | 2.1384 |
| Final Perplexity | 8.4861 |
| Best Checkpoint | Epoch 5 |
| Best KV-cache Speedup | 1.5754x |

### Experiment 02C Epoch Results

| Epoch | Optimizer Step | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|---:|
| 1 | 420 | 4.1268 | 3.5834 | 35.9957 |
| 2 | 840 | 3.0810 | 2.6652 | 14.3712 |
| 3 | 1260 | 2.4780 | 2.3225 | 10.2008 |
| 4 | 1680 | 2.2166 | 2.1791 | 8.8383 |
| 5 | 2100 | 2.1006 | 2.1384 | 8.4861 |

### Attention Comparison

| Metric | Experiment 02B: Standard Attention | Experiment 02C: Flash/SDPA Attention |
|---|---:|---:|
| Final Train Loss | 2.0879 | 2.1006 |
| Final Validation Loss | 2.1301 | 2.1384 |
| Final Perplexity | 8.4161 | 8.4861 |
| Best KV-cache Speedup | 1.4896x | 1.5754x |

Experiment 02C preserved nearly the same validation quality as Experiment 02B while achieving the strongest KV-cache speedup among the tested configurations.

---

## Final Experiment Comparison

| Experiment | Tokenizer | Attention | Train Loss | Validation Loss | Perplexity | Best KV Speedup |
|---|---|---|---:|---:|---:|---:|
| Experiment 01 | GPT-2 BPE / `tiktoken` | Standard | 0.2834 | 11.9652 | 157,185 | Not tested |
| Experiment 02A | GPT-2 BPE / `tiktoken` | Standard | 3.3203 | 4.1917 | 66.1369 | 1.5389x |
| Experiment 02B | Custom biomedical BPE | Standard | 2.0879 | 2.1301 | 8.4161 | 1.4896x |
| Experiment 02C | Custom biomedical BPE | Flash/SDPA | 2.1006 | 2.1384 | 8.4861 | 1.5754x |

---

## Main Findings

### 1. Experiment 01 validated the pipeline but overfit.

The first experiment confirmed that the model, tokenizer, training loop, validation loop, and checkpoint flow worked correctly. However, the dataset was too small, and the model overfit severely.

### 2. Experiment 02A fixed the baseline setup.

Experiment 02A improved the training setup using a larger medical-domain corpus, longer context length, larger model configuration, better checkpointing, and KV-cache benchmarking.

### 3. Custom biomedical BPE significantly improved medical-domain modeling.

Experiment 02B reduced validation loss from `4.1917` to `2.1301` and perplexity from `66.1369` to `8.4161` compared with Experiment 02A.

### 4. Flash/SDPA attention preserved model quality.

Experiment 02C achieved a final validation loss of `2.1384`, very close to Experiment 02B's `2.1301`, showing that the optimized attention path preserved model quality.

### 5. KV-cache improved inference most strongly at longer prompts.

Across experiments, KV-cache showed the strongest speedups at prompt length 512.

---

## Model Configuration Used in Final Experiment Series

| Parameter | Value |
|---|---:|
| Context Length | 1024 |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Feedforward Hidden Dimension | 3072 |
| Dropout | 0.1 |
| QKV Bias | False |
| Batch Size | 8 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 64 |
| Approximate Model Size | ~194M parameters |

The final model size is approximately 194M parameters due to the vocabulary size, untied embedding/LM head weights, and the feedforward configuration.

---

## Running Experiment 02A

```bash
python -m training.train_experiment_02a \
  --data_dir /path/to/text_chunks \
  --num_epochs 5 \
  --batch_size 8
```

---

## Running Experiment 02B

```bash
python -m training.train_experiment_02b \
  --tokenized_dir /path/to/tokenized_chunks \
  --num_epochs 5 \
  --batch_size 8
```

---

## Running Experiment 02C

```bash
python -m training.train_experiment_02c \
  --tokenized_dir /path/to/tokenized_chunks \
  --num_epochs 5 \
  --batch_size 8
```

---

## KV-Cache Benchmark

Example benchmark command:

```bash
python -m inference.benchmark_kv_cache \
  --config experiments/exp_02b_custom_bpe_medical_5m/config.json \
  --checkpoint experiments/exp_02b_custom_bpe_medical_5m/checkpoints/best.pt \
  --output experiments/exp_02b_custom_bpe_medical_5m/kv_cache_benchmark.csv \
  --prompt_lengths 64,128,256,512 \
  --max_new_tokens 64
```

---

## Checkpointing

The final training pipeline saves:

```text
best.pt
latest.pt
epoch_N.pt
config.json
train_log.csv
val_log.csv
kv_cache_benchmark.csv
```

Large checkpoint files are excluded from GitHub and backed up separately using external storage such as Google Drive.

---

## Optional: Google Drive Backup with rclone

For cloud training environments, `rclone` can be used to back up experiment outputs and checkpoints to Google Drive.

Install `rclone`:

```bash
curl https://rclone.org/install.sh | sudo bash
```

Configure Google Drive:

```bash
rclone config
```

Example backup command:

```bash
rclone copy experiments/exp_02a_tiktoken_medical_5m \
  gdrive:LLM_results/exp_02a_tiktoken_medical_5m \
  --progress
```

`rclone` is optional and should not be added to `requirements.txt`.

---

## Documentation

Each experiment has its own documentation folder containing:

```text
README.md
Experiment report PDF
plots/
```

The documentation folders are committed to GitHub for reviewability, while runtime training outputs and large checkpoints are kept outside Git or backed up separately.

---

## Future Work

Future work will focus on scaling the project beyond the controlled Experiment 02 series.

Planned directions include:

- Training on a larger medical-domain corpus of approximately 1B tokens
- Rebuilding the custom biomedical BPE tokenizer in Rust
- Improving tokenizer training speed and memory efficiency
- Extending training to larger model and data scales
- Evaluating longer context windows
- Adding more detailed generation-quality evaluation
- Expanding throughput and memory profiling
- Preparing the model pipeline for production-oriented serving workflows

A Rust-based tokenizer implementation is especially useful for billion-token-scale preprocessing because it can improve speed, memory efficiency, and scalability.

---

## Conclusion

This project demonstrates a complete from-scratch GPT-style language modeling pipeline.

The work progressed from a small WikiText baseline to a controlled medical-domain experiment series comparing tokenizer quality, attention implementation, and KV-cache inference behavior.

The strongest modeling improvement came from the custom biomedical BPE tokenizer in Experiment 02B. The strongest KV-cache speedup appeared in Experiment 02C, where Flash/SDPA attention preserved model quality while improving inference efficiency.

Overall, the project shows the importance of tokenizer design, reproducible training infrastructure, checkpointing, and inference benchmarking in building domain-specific GPT-style language models from scratch.