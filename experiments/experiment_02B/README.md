# Experiment 02B — Medical GPT Training using Custom Biomedical BPE

## Overview

Experiment 02B is the tokenizer-focused follow-up to Experiment 02A. It trains the same GPT-style decoder-only language model setup on the medical-domain corpus, but replaces the generic GPT-2/tiktoken tokenizer with a custom biomedical BPE tokenizer.

The purpose of this experiment is to evaluate whether domain-specific tokenization improves medical language modeling performance while keeping the overall training setup comparable to Experiment 02A.

This experiment was conducted to:

- Train a GPT-style decoder-only Transformer using a custom biomedical BPE tokenizer
- Evaluate medical-domain language modeling performance
- Compare tokenizer behavior against the Experiment 02A GPT-2/tiktoken baseline
- Measure training loss, validation loss, and perplexity across epochs
- Verify stable checkpoint preservation using `best.pt`, `latest.pt`, and epoch checkpoints
- Benchmark autoregressive inference with and without KV-cache

---

## Experiment Scope

Experiment 02B focuses on tokenizer quality.

The main change from Experiment 02A is the tokenizer. Experiment 02A used the GPT-2 BPE tokenizer through `tiktoken`, while Experiment 02B uses a custom biomedical BPE tokenizer trained for medical-domain text.

| Experiment | Tokenizer | Attention | Purpose |
|---|---|---|---|
| Experiment 02A | GPT-2 BPE through `tiktoken` | Standard causal attention | Medical-domain baseline |
| Experiment 02B | Custom biomedical BPE | Standard causal attention | Tokenizer comparison |

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 02B |
| Dataset | Medical text corpus |
| Dataset Format | Pre-tokenized chunks |
| Tokenizer | Custom biomedical BPE |
| Vocabulary Size | 52,000 |
| Model Type | GPT-style decoder-only Transformer |
| Attention Type | Standard causal multi-head self-attention |
| Hardware | Lightning AI H200 |
| Epochs | 5 |
| Batch Size | 8 |
| Gradient Accumulation | 8 steps |
| Effective Batch Size | 64 |
| Context Length | 1024 |
| Objective | Medical-domain tokenizer comparison |
| Benchmark | KV-cache inference benchmark |

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 52,000 |
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

### 3. Prepare Tokenized Medical Chunks

Experiment 02B expects tokenized medical chunks generated using the custom biomedical BPE tokenizer.

The final training run used tokenized chunks copied into the following structure:

```text
exp_02b_custom_bpe/
└── tokenized_chunks/
```

Example path used during final training:

```text
/teamspace/studios/this_studio/exp_02b_custom_bpe/tokenized_chunks
```

### 4. Train the Model

Run Experiment 02B using the dedicated training entry point:

```bash
python -m training.train_experiment_02b \
  --tokenized_dir /teamspace/studios/this_studio/exp_02b_custom_bpe/tokenized_chunks \
  --num_epochs 5 \
  --batch_size 8
```

For a local setup, replace the `--tokenized_dir` path with the location of the prepared custom-BPE tokenized chunks.

---

## Outputs

Experiment outputs are stored in:

```text
experiments/exp_02b_custom_bpe_medical_5m/
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

| Epoch | Optimizer Step | Train Loss | Validation Loss | Perplexity | Learning Rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 420 | 4.122820 | 3.588786 | 36.1901 | 2.8382e-4 |
| 2 | 840 | 3.043306 | 2.642536 | 14.0488 | 2.1493e-4 |
| 3 | 1260 | 2.464203 | 2.313835 | 10.1131 | 1.2030e-4 |
| 4 | 1680 | 2.204899 | 2.170586 | 8.7634 | 4.0600e-5 |
| 5 | 2100 | 2.087899 | 2.130147 | 8.4161 | 1.0000e-5 |

---

## Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 2.0879 |
| Final Validation Loss | 2.1301 |
| Final Perplexity | 8.4161 |
| Best Checkpoint | Epoch 5 |
| Best Checkpoint File | `best.pt` |

Experiment 02B showed strong convergence and a major improvement over the GPT-2/tiktoken baseline from Experiment 02A.

Validation loss decreased from `3.5888` to `2.1301`, and perplexity decreased from `36.1901` to `8.4161`.

---

## Comparison with Experiment 02A

| Metric | Experiment 02A: GPT-2/tiktoken | Experiment 02B: Custom Biomedical BPE |
|---|---:|---:|
| Final Train Loss | 3.3203 | 2.0879 |
| Final Validation Loss | 4.1917 | 2.1301 |
| Final Perplexity | 66.1369 | 8.4161 |

Experiment 02B significantly improved validation loss and perplexity compared with Experiment 02A.

The key change was the tokenizer. This suggests that the custom biomedical BPE tokenizer produced more suitable token representations for the medical corpus than the generic GPT-2/tiktoken tokenizer.

---

## KV-Cache Benchmark

After training, Experiment 02B was evaluated using an autoregressive inference benchmark with and without KV-cache.

### Benchmark Command

```bash
python -m inference.benchmark_kv_cache \
  --config experiments/exp_02b_custom_bpe_medical_5m/config.json \
  --checkpoint experiments/exp_02b_custom_bpe_medical_5m/checkpoints/best.pt \
  --output experiments/exp_02b_custom_bpe_medical_5m/kv_cache_benchmark.csv \
  --prompt_lengths 64,128,256,512 \
  --max_new_tokens 64
```

### KV-Cache Latency and Speedup

| Prompt Length | Max New Tokens | Latency Without KV | Latency With KV | Speedup |
|---:|---:|---:|---:|---:|
| 64 | 64 | 0.339639 | 0.308528 | 1.1008x |
| 128 | 64 | 0.351796 | 0.310532 | 1.1329x |
| 256 | 64 | 0.338507 | 0.311260 | 1.0875x |
| 512 | 64 | 0.452341 | 0.303662 | 1.4896x |

### KV-Cache Throughput and Memory

| Prompt Length | Tokens/sec Without KV | Tokens/sec With KV | Peak Memory Without KV | Peak Memory With KV |
|---:|---:|---:|---:|---:|
| 64 | 188.45 | 207.45 | 823.13 MB | 795.98 MB |
| 128 | 182.92 | 206.13 | 848.32 MB | 829.60 MB |
| 256 | 189.07 | 205.62 | 900.70 MB | 867.82 MB |
| 512 | 141.49 | 210.76 | 1001.83 MB | 975.28 MB |

---

## KV-Cache Analysis

KV-cache improved inference speed across all tested prompt lengths.

The strongest speedup appeared at prompt length `512`, where latency decreased from `0.452341s` to `0.303662s`, producing a `1.4896x` speedup. Throughput increased from `141.49` tokens/sec to `210.76` tokens/sec.

This confirms that KV-cache is most useful at longer prompt lengths, where cached key-value states reduce repeated attention computation during autoregressive generation.

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

Experiment 02B demonstrated that tokenizer choice has a strong impact on medical-domain language modeling performance.

Compared with Experiment 02A, the custom biomedical BPE tokenizer produced substantially lower validation loss and perplexity. This indicates that medical-domain subword segmentation helped the model represent biomedical terminology more effectively.

The training process remained stable across all five epochs. Validation loss decreased consistently, and the final perplexity reached `8.4161`, making Experiment 02B a strong improvement over the GPT-2/tiktoken baseline.

The KV-cache benchmark also confirmed that inference efficiency improved when cached decoding was used, especially for longer prompts.

---

## Future Work

The next stage can extend this experiment by keeping the same medical-domain setting while exploring efficiency-oriented improvements.

At a high level, the next experiment should investigate whether the model can preserve the quality gains from the custom biomedical tokenizer while improving training or inference efficiency through implementation-level optimization.

This future direction keeps the focus on controlled experimentation: first improving tokenizer quality, then studying whether the same modeling behavior can be maintained under a more efficient attention implementation.

---

## Conclusion

Experiment 02B successfully established a strong custom-tokenizer medical GPT baseline.

The experiment achieved a final validation loss of `2.1301` and a final perplexity of `8.4161`, substantially improving over the GPT-2/tiktoken baseline from Experiment 02A.

The results support the conclusion that a custom biomedical BPE tokenizer is significantly better suited for medical-domain language modeling than a generic tokenizer. The KV-cache benchmark further confirmed that cached autoregressive decoding improves inference throughput, especially at longer prompt lengths.

Overall, Experiment 02B provides strong evidence that domain-specific tokenization improves language modeling quality on biomedical text.