import torch
import torch.nn as nn

from model.embeddings import GPTEmbeddings
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.max_len = config.context_length
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_len=config.context_length,
            dropout=config.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    bias=config.qkv_bias,
                    ffn_hidden_dim=config.ffn_hidden_dim,
                    attention_type=config.attention_type,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = LayerNorm(
            d_model=config.d_model,
            bias=config.qkv_bias,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids shape (batch_size, seq_len), "
                f"but got {tuple(input_ids.shape)}"
            )

        _, T = input_ids.shape

        if past_kv is None:
            past_length = 0
        else:
            if len(past_kv) != self.num_layers:
                raise ValueError(
                    f"Expected past_kv for {self.num_layers} layers, "
                    f"but got {len(past_kv)}"
                )

            past_length = past_kv[0][0].shape[2]

        total_length = past_length + T

        if total_length > self.max_len:
            raise ValueError(
                f"Sequence length {total_length} exceeds max_len {self.max_len}"
            )

        x = self.embeddings(
            input_ids,
            start_pos=past_length,
        )

        present_kv = [] if use_cache else None

        for idx, block in enumerate(self.blocks):

            layer_past = None

            if past_kv is not None:
                layer_past = past_kv[idx]

            if use_cache:
                x, layer_present = block(
                    x,
                    past_kv=layer_past,
                    use_cache=True,
                )

                present_kv.append(layer_present)

            else:
                x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        if use_cache:
            return logits, present_kv

        return logits

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()

        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)

        return sum(p.numel() for p in parameters)