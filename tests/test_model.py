import torch

from config.config import GPTConfig
from model.gpt_model import GPTModel
from training.loss import GPTLoss


def test_gpt_model_forward_and_loss():

    config = GPTConfig(
        vocab_size=1000,
        context_length=128,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        qkv_bias=False,
    )

    model = GPTModel(config)

    criterion = GPTLoss()

    batch_size = 2
    seq_len = 64

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
    )

    logits = model(input_ids)

    assert logits.shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    )

    loss = criterion(logits, targets)

    assert loss.ndim == 0

    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

