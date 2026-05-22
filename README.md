# LLM from Scratch

A from-scratch implementation of a GPT-style decoder-only Transformer in PyTorch, accompanied by a custom biomedical Byte Pair Encoding (BPE) tokenizer and a series of controlled language model training experiments.

---

## Project Highlights

- GPT-style decoder-only Transformer implementation
- Custom Biomedical BPE Tokenizer (~52K vocabulary)
- Multi-Head Causal Self-Attention
- Layer Normalization and Residual Connections
- Mixed Precision Training (AMP)
- Gradient Accumulation and Gradient Clipping
- Warmup + Cosine Learning Rate Scheduling
- Validation Loss and Perplexity Evaluation
- Checkpoint Saving and Best Model Tracking
- Autoregressive Text Generation
- Controlled Tokenizer Comparison Experiments

---

## Repository Structure

```text
LLM-from-Scratch/
│
├── config/
├── data/
├── model/
│   └── attention/
├── training/
├── inference/
├── utils/
├── tests/
│
├── experiments/
│   ├── experiment_01/
│   ├── experiment_02A/
│   └── experiment_02B/
│
├── resources/
├── scripts/
├── checkpoints/
└── README.md
```

---

## Model Architecture

| Component | Configuration |
|------------|------------|
| Architecture | Decoder-only Transformer |
| Attention | Multi-Head Causal Self-Attention |
| Feedforward | Position-wise Feedforward Network |
| Normalization | Layer Normalization |
| Positional Encoding | Learned Positional Embeddings |
| Objective | Next Token Prediction |
| Framework | PyTorch |

---

## Experimental Progression

### Experiment 01 — GPT Training Baseline using WikiText-103

| Component | Value |
|------------|------------|
| Dataset | WikiText-103 Subset |
| Tokenizer | GPT-2 TikToken |
| Objective | Training Pipeline Validation |

Objectives:

- Validate end-to-end training pipeline
- Test tokenizer integration
- Verify checkpoint saving and loading
- Establish baseline training metrics

---

### Experiment 02A — Medical GPT Training using GPT-2 TikToken

| Component | Value |
|------------|------------|
| Dataset | Biomedical Corpus |
| Dataset Size | ~5M Tokens |
| Tokenizer | GPT-2 TikToken |
| Objective | Medical Language Modeling |

Final Results:

| Metric | Value |
|------------|------------|
| Train Loss | 3.1951 |
| Validation Loss | 4.0208 |
| Perplexity | 55.7466 |

---

### Experiment 02B — Medical GPT Training using Custom Biomedical BPE Tokenizer

| Component | Value |
|------------|------------|
| Dataset | Biomedical Corpus |
| Dataset Size | ~5M Tokens |
| Tokenizer | Custom Biomedical BPE |
| Vocabulary Size | ~52K |
| Objective | Medical Language Modeling |

Final Results:

| Metric | Value |
|------------|------------|
| Train Loss | 1.9890 |
| Validation Loss | 2.0155 |
| Perplexity | 7.5042 |

Comparison with Experiment 02A:

| Metric | GPT-2 TikToken | Custom BPE |
|------------|------------|------------|
| Validation Loss | 4.0208 | 2.0155 |
| Perplexity | 55.7466 | 7.5042 |

Key Findings:

- Reduced token fragmentation
- Improved token efficiency
- Faster convergence
- Lower validation loss
- Lower perplexity
- Better biomedical representation

---

## Training Features

| Feature | Status |
|------------|------------|
| Mixed Precision Training (AMP) | ✓ |
| Gradient Accumulation | ✓ |
| Gradient Clipping | ✓ |
| AdamW Optimizer | ✓ |
| Warmup Scheduling | ✓ |
| Cosine Learning Rate Decay | ✓ |
| Validation Evaluation | ✓ |
| Perplexity Calculation | ✓ |
| Checkpoint Saving | ✓ |
| Best Model Tracking | ✓ |

---

## Inference Features

- Autoregressive Text Generation
- Temperature Sampling
- Top-k Sampling
- Checkpoint Loading
- Model Evaluation Utilities

---

## Future Work — Experiment 03A

Planned large-scale biomedical language model training experiment.

| Component | Target |
|------------|------------|
| Dataset Size | ~30–40M Tokens |
| Tokenizer | Custom Biomedical BPE |
| Vocabulary Size | ~52K |
| Context Length | 1024 |
| Embedding Dimension | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Model Size | ~124M Parameters |

Additional Evaluation:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- KV Cache Benchmarking
- Throughput Analysis
- Memory Profiling

---

## Related Repository

### Domain-Specific-BPE-Tokenizer

Custom biomedical Byte Pair Encoding tokenizer developed for domain-specific language model training.

Key Capabilities:

- BPE Merge Learning
- Vocabulary Construction
- Encoding and Decoding
- Serialization Support
- Modular Architecture
- Biomedical Corpus Training

---

## Author

**Adithya Prabhu**

Implementation and experimental evaluation of biomedical tokenization and GPT-style language modeling systems.