Building a Domain-Specific Language Modeling Pipeline from Scratch
Custom Biomedical Tokenization and GPT Training Research

Author: Adithya Prabhu
Project Duration: 2025–2026
Repositories:

Domain-Specific-BPE-Tokenizer
LLM-from-Scratch
Abstract

This project presents the design, implementation, and evaluation of a complete language modeling pipeline developed entirely from scratch. The work consists of two major components: a custom biomedical Byte Pair Encoding (BPE) tokenizer and a GPT-style decoder-only Transformer language model implemented in PyTorch without relying on external transformer frameworks.

The tokenizer project focused on constructing a domain-specific subword vocabulary capable of efficiently representing biomedical terminology. The language model project focused on implementing the complete transformer training pipeline, including embeddings, positional encodings, causal self-attention, feedforward networks, training infrastructure, evaluation metrics, checkpointing, mixed precision training, and autoregressive text generation.

Three consecutive experimental phases were designed. Experiment 01 validated the end-to-end training infrastructure on WikiText-103. Experiment 02A established a medical-domain baseline using GPT-2 TikToken tokenization. Experiment 02B replaced TikToken with the custom biomedical tokenizer while preserving all model and training configurations. Results demonstrated substantial improvements in validation loss and perplexity when using domain-specific tokenization.

The project establishes a foundation for large-scale language model training planned in Experiment 03A, where the custom tokenizer will be combined with a 124M-parameter GPT architecture trained on hundreds of millions of biomedical tokens.

1. Introduction

Modern large language models rely on two fundamental components:

Efficient tokenization
Transformer-based sequence modeling

Although most practitioners use pre-built tokenizers and transformer libraries, understanding the internal mechanics of language models requires implementing these systems from first principles.

The primary objective of this project was therefore to design and implement an end-to-end language modeling pipeline without using external transformer frameworks while maintaining reproducibility, modularity, and experimental rigor.

The work was divided into two independent yet interconnected projects:

Domain-Specific BPE Tokenizer
LLM-from-Scratch

Together these systems form a complete training stack capable of processing raw biomedical text and producing autoregressive language models.

2. Project Architecture

The complete pipeline consists of:

Raw Biomedical Text
↓
Corpus Preparation
↓
Custom BPE Tokenizer Training
↓
Vocabulary Construction
↓
Tokenization
↓
Dataset Chunking
↓
GPT Training Pipeline
↓
Evaluation
↓
Checkpointing
↓
Inference and Text Generation

This modular architecture enables independent experimentation with tokenization strategies while maintaining identical model configurations.

3. Domain-Specific BPE Tokenizer Project
Motivation

General-purpose tokenizers such as GPT-2 BPE were trained on broad internet text and often fragment specialized biomedical terminology into multiple subword units.

Examples include:

pharmacokinetics
neurodegeneration
cardiomyopathy
immunohistochemistry

Fragmentation increases sequence length and may reduce learning efficiency.

The objective was therefore to construct a tokenizer trained directly on biomedical text.

Core Features

The tokenizer implementation includes:

Byte-level vocabulary initialization
Iterative BPE merge learning
Vocabulary construction
Ranked merge tracking
Encoding pipeline
Decoding pipeline
Serialization support
Model loading and saving
Unknown token handling
Modular architecture
Software Architecture

Major modules include:

trainer.py
encoder.py
decoder.py
vocab.py
serialization.py
bpe_tokenizer.py

The architecture was intentionally separated into independent components to improve maintainability and testing.

Testing Framework

Comprehensive testing was implemented for:

Encoding correctness
Decoding correctness
Serialization integrity
Merge application
Vocabulary construction
End-to-end consistency

All tokenizer outputs satisfy:

Decode(Encode(Text)) = Text

ensuring lossless reconstruction.

Final Tokenizer

Final biomedical tokenizer characteristics:

Component	Value
Vocabulary Size	~52,000
Tokenization Type	Byte Pair Encoding
Domain	Biomedical
Serialization	JSON
Language	Python
Architecture	Modular
4. GPT Language Model Project
Objective

The second project focused on implementing a GPT-style autoregressive language model from scratch.

The implementation reproduces the major architectural components of GPT-style decoder-only transformers while remaining fully understandable and extensible.

Implemented Components
Token Embeddings

Learned token representations convert discrete token identifiers into dense vector spaces.

Positional Embeddings

Explicit positional information is injected into token representations to preserve sequence order.

Multi-Head Causal Self-Attention

The attention mechanism enables each token to attend to previous context while enforcing autoregressive masking.

Feedforward Networks

Each transformer block contains position-wise nonlinear transformations to improve representational capacity.

Residual Connections

Residual pathways improve optimization stability in deep networks.

Layer Normalization

Normalization improves gradient flow and convergence.

Language Modeling Head

The final projection layer maps hidden representations back into vocabulary probabilities.

Training Infrastructure

Implemented features include:

Mixed Precision Training
Gradient Accumulation
Gradient Clipping
AdamW Optimization
Cosine Learning Rate Scheduling
Warmup Scheduling
Validation Evaluation
Perplexity Computation
Checkpoint Saving
Best Model Tracking
Text Generation Pipeline
5. Experiment 01
Objective

Validate the complete training pipeline before conducting domain-specific experiments.

Dataset:

WikiText-103 subset

Tokenizer:

GPT-2 TikToken

Results

Experiment 01 successfully verified:

Dataset loading
Tokenization
Transformer training
Validation evaluation
Checkpoint saving
Text generation

This experiment established the baseline infrastructure used throughout subsequent experiments.

6. Experiment 02A
Objective

Train a medical-domain GPT model using GPT-2 TikToken.

Dataset:

Biomedical corpus (~5 million tokens)

Tokenizer:

GPT-2 TikToken

Results
Epoch	Train Loss	Validation Loss	Perplexity
1	5.4626	4.6626	105.9074
5	3.1951	4.0208	55.7466

Validation loss decreased consistently throughout training.

The experiment established the tokenizer baseline used for comparison against the custom tokenizer.

7. Experiment 02B
Objective

Evaluate the impact of replacing GPT-2 TikToken with the custom biomedical tokenizer while keeping all other training variables fixed.

Controlled variables:

Dataset
Architecture
Hyperparameters
Training schedule
Optimizer

Only tokenizer changed.

Results
Epoch	Train Loss	Validation Loss	Perplexity
1	3.4282	2.7003	14.8835
5	1.9890	2.0155	7.5042
Comparison Against Experiment 02A
Metric	Experiment 02A	Experiment 02B
Tokenizer	GPT-2 TikToken	Custom Biomedical BPE
Final Train Loss	3.1951	1.9890
Final Validation Loss	4.0208	2.0155
Final Perplexity	55.7466	7.5042
Key Findings

The custom tokenizer:

Reduced medical term fragmentation
Improved token efficiency
Accelerated convergence
Lowered validation loss
Reduced perplexity
Improved generalization

Experiment 02B validated the effectiveness of domain-specific tokenization for biomedical language modeling.

8. Research Contributions

The project demonstrates:

Contribution 1

Development of a fully modular biomedical BPE tokenizer from scratch.

Contribution 2

Implementation of a GPT-style Transformer architecture without external transformer libraries.

Contribution 3

Controlled tokenizer comparison using identical training conditions.

Contribution 4

Empirical evidence supporting domain-specific tokenization for biomedical language modeling.

9. Future Work — Experiment 03A

Experiment 03A will scale both tokenizer and model training substantially.

Planned configuration:

Component	Target
Vocabulary	~100,000
Model Parameters	~124 Million
Dataset Size	~600 Million Tokens
Architecture	GPT-style Decoder Transformer
Training Objective	Large-scale Biomedical Language Modeling

Expected additions:

Larger context windows
Expanded tokenizer vocabulary
Improved scaling behavior
Extended evaluation framework
Detailed convergence analysis
Large-scale checkpoint management

Experiment 03A represents the transition from tokenizer validation toward large-scale language model training.

10. Conclusion

This work demonstrates the successful construction of an end-to-end language modeling stack from scratch, consisting of a custom biomedical tokenizer and a GPT-style Transformer implementation.

Through three consecutive experimental stages, the project progressed from infrastructure validation to controlled tokenizer research. Results showed that domain-specific tokenization significantly improved validation loss and perplexity under identical training conditions, validating the effectiveness of the custom biomedical BPE tokenizer.

The completion of Experiment 02B establishes a strong foundation for Experiment 03A, where the tokenizer and model will be scaled to substantially larger training regimes. Collectively, these projects represent a comprehensive study of tokenization, transformer architectures, optimization strategies, and large-scale language model training from first principles.