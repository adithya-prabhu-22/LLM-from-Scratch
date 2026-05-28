from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 128
    stride: int = 64

    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4

    dropout: float = 0.1
    qkv_bias: bool = False

    ffn_hidden_dim: int | None = 1024

    attention_type: str = "standard"

    max_new_tokens: int = 30
    temperature: float = 0.8
    top_k: int | None = 40

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1

    grad_clip: float = 1.0
    use_amp: bool = True
    gradient_accumulation_steps: int = 4

    batch_size: int = 2
    num_epochs: int = 1

    max_train_batches: int | None = 50
    max_val_batches: int | None = 10

    save_every_epoch: bool = True
    save_best_checkpoint: bool = True