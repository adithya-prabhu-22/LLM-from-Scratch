# Experiment 02C — Medical GPT Training using Custom Biomedical BPE and Flash Attention

## Overview

Experiment 02C is the efficiency-focused continuation of Experiment 02B. It uses the same medical-domain corpus and custom biomedical BPE tokenizer, but replaces the standard attention path with a Flash/SDPA attention implementation.

The purpose of this experiment is to evaluate whether the model can preserve the strong medical-domain language modeling performance achieved in Experiment 02B while improving systems-level efficiency during training and inference.

This experiment was conducted to:

- Train a GPT-style decoder-only Transformer using a custom biomedical BPE tokenizer
- Evaluate Flash/SDPA attention in the medical-domain GPT training pipeline
- Compare training quality against the Experiment 02B standard-attention baseline
- Measure training loss, validation loss, and perplexity across epochs
- Verify stable checkpoint preservation using `best.pt`, `latest.pt`, and epoch checkpoints
- Benchmark autoregressive inference with and without KV-cache
- Analyze whether optimized attention preserves model quality while improving throughput

---

## Experiment Scope

Experiment 02C focuses on attention efficiency.

Experiment 02B used the custom biomedical BPE tokenizer with standard causal attention. Experiment 02C keeps the custom biomedical BPE tokenizer but switches the attention implementation to Flash/SDPA attention.

| Experiment | Tokenizer | Attention | Purpose |
|---|---|---|---|
| Experiment 02B | Custom biomedical BPE | Standard causal attention | Tokenizer-focused baseline |
| Experiment 02C | Custom biomedical BPE | Flash/SDPA attention | Attention efficiency comparison |

---

## Experiment Summary

| Component | Value |
|---|---|
| Experiment | Experiment 02C |
| Dataset | Medical text corpus |
| Dataset Format | Pre-tokenized chunks |
| Tokenizer | Custom biomedical BPE |
| Vocabulary Size | 52,000 |
| Model Type | GPT-style decoder-only Transformer |
| Attention Type | Flash/SDPA causal attention |
| Hardware | Lightning AI H200 |
| Epochs | 5 |
| Batch Size | 8 |
| Gradient Accumulation | 8 steps |
| Effective Batch Size | 64 |
| Context Length | 1024 |
| Objective | Attention efficiency evaluation |
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
| Attention Type | Flash/SDPA causal attention |
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

## Running Experiment 02C

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

Experiment 02C expects tokenized medical chunks generated using the custom biomedical BPE tokenizer.

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

Run Experiment 02C using the dedicated training entry point:

```bash
python -m training.train_experiment_02c \
  --tokenized_dir /teamspace/studios/this_studio/exp_02b_custom_bpe/tokenized_chunks \
  --num_epochs 5 \
  --batch_size 8
```

For a local setup, replace the `--tokenized_dir` path with the location of the prepared custom-BPE tokenized chunks.

---

## Outputs

Experiment outputs are stored in:

```text
experiments/exp_02c_custom_bpe_flash_attention_medical_5m/
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
| 1 | 420 | 4.126836 | 3.583400 | 35.9957 | 2.8382e-4 |
| 2 | 840 | 3.080969 | 2.665223 | 14.3712 | 2.1493e-4 |
| 3 | 1260 | 2.478034 | 2.322465 | 10.2008 | 1.2030e-4 |
| 4 | 1680 | 2.216577 | 2.179097 | 8.8383 | 4.0600e-5 |
| 5 | 2100 | 2.100648 | 2.138426 | 8.4861 | 1.0000e-5 |

---

## Final Result

| Metric | Value |
|---|---:|
| Final Train Loss | 2.1006 |
| Final Validation Loss | 2.1384 |
| Final Perplexity | 8.4861 |
| Best Checkpoint | Epoch 5 |
| Best Checkpoint File | `best.pt` |

Experiment 02C showed stable convergence across all five epochs. Validation loss decreased from `3.5834` to `2.1384`, and perplexity decreased from `35.9957` to `8.4861`.

The final result remained close to Experiment 02B, showing that Flash/SDPA attention preserved the model quality achieved by the custom biomedical tokenizer.

---

## Comparison with Experiment 02B

| Metric | Experiment 02B: Standard Attention | Experiment 02C: Flash/SDPA Attention |
|---|---:|---:|
| Final Train Loss | 2.0879 | 2.1006 |
| Final Validation Loss | 2.1301 | 2.1384 |
| Final Perplexity | 8.4161 | 8.4861 |

Experiment 02C achieved nearly the same validation performance as Experiment 02B.

The small difference in validation loss and perplexity suggests that the Flash/SDPA attention path preserved the language modeling behavior of the standard-attention model while enabling a more optimized attention implementation.

---

## KV-Cache Benchmark

After training, Experiment 02C was evaluated using an autoregressive inference benchmark with and without KV-cache.

### Benchmark Command

```bash
python -m inference.benchmark_kv_cache \
  --config experiments/exp_02c_custom_bpe_flash_attention_medical_5m/config.json \
  --checkpoint experiments/exp_02c_custom_bpe_flash_attention_medical_5m/checkpoints/best.pt \
  --output experiments/exp_02c_custom_bpe_flash_attention_medical_5m/kv_cache_benchmark.csv \
  --prompt_lengths 64,128,256,512 \
  --max_new_tokens 64
```

### KV-Cache Latency and Speedup

| Prompt Length | Max New Tokens | Latency Without KV | Latency With KV | Speedup |
|---:|---:|---:|---:|---:|
| 64 | 64 | 0.290073 | 0.279894 | 1.0364x |
| 128 | 64 | 0.289977 | 0.282696 | 1.0258x |
| 256 | 64 | 0.296323 | 0.286467 | 1.0344x |
| 512 | 64 | 0.441901 | 0.280507 | 1.5754x |

### KV-Cache Throughput and Memory

| Prompt Length | Tokens/sec Without KV | Tokens/sec With KV | Peak Memory Without KV | Peak Memory With KV |
|---:|---:|---:|---:|---:|
| 64 | 220.63 | 228.66 | 823.13 MB | 795.98 MB |
| 128 | 220.71 | 226.39 | 848.32 MB | 829.60 MB |
| 256 | 216.20 | 223.45 | 900.70 MB | 867.82 MB |
| 512 | 144.83 | 228.17 | 1001.83 MB | 975.28 MB |

---

## KV-Cache Analysis

KV-cache improved inference speed across all tested prompt lengths.

The strongest speedup appeared at prompt length `512`, where latency decreased from `0.441901s` to `0.280507s`, producing a `1.5754x` speedup. Throughput increased from `144.83` tokens/sec to `228.17` tokens/sec.

This confirms that KV-cache provides the most benefit at longer prompt lengths, where cached key-value states reduce repeated attention computation during autoregressive generation.

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

Experiment 02C demonstrated that the optimized Flash/SDPA attention path can preserve the strong medical-domain modeling quality achieved in Experiment 02B.

The final validation loss of `2.1384` and final perplexity of `8.4861` are very close to the Experiment 02B standard-attention result. This indicates that the attention implementation change did not significantly harm language modeling quality.

The KV-cache benchmark also showed strong inference improvements at longer prompt lengths. The best observed speedup was `1.5754x` at prompt length `512`, making Experiment 02C the strongest KV-cache speedup result among the tested configurations.

Overall, Experiment 02C supports the conclusion that systems-level attention optimization can be introduced while maintaining the benefits of domain-specific biomedical tokenization.

---

## Future Work

Future work can extend this experiment by evaluating additional systems-level optimizations while preserving the same controlled medical-domain setting.

Possible future directions include:

- Larger medical-domain corpora
- Longer context windows
- Additional tokenizer vocabulary-size ablations
- More detailed throughput and memory profiling
- Generation-quality evaluation using medical-domain prompts
- Serving-oriented benchmarking for deployment scenarios

These extensions would help determine how well the current training and inference pipeline scales beyond the controlled Experiment 02 series.

---

## Conclusion

Experiment 02C successfully evaluated the custom biomedical BPE medical GPT model using a Flash/SDPA attention path.

The experiment achieved a final validation loss of `2.1384` and a final perplexity of `8.4861`, closely matching the standard-attention Experiment 02B result. This shows that the optimized attention implementation preserved model quality.

The KV-cache benchmark confirmed that cached autoregressive decoding improves inference throughput, especially for longer prompt lengths. The strongest observed speedup was `1.5754x` at prompt length `512`.

Overall, Experiment 02C completes the controlled Experiment 02 series by showing that domain-specific tokenization can be combined with attention-level optimization while maintaining stable training behavior and strong inference efficiency.