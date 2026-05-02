from dataclasses import dataclass

@dataclass
class GPTConfig:

    vocab_size: int = 32000
    context_length: int = 1024

    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12

    dropout: float = 0.1
    qkv_bias: bool = False

    ffn_hidden_dim: int | None = None

    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int | None = 40

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5

    grad_clip: float = 1.0

    use_amp: bool = True

    gradient_accumulation_steps: int = 4

