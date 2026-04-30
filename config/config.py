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
    ffn_hidden_dim: int = 4 * 768