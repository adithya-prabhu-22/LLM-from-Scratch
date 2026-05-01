import torch
import tiktoken
from torch.optim import AdamW

from config.config import GPTConfig
from model.gpt_model import GPTModel
from training.loss import GPTLoss
from data.dataloader import create_dataloader
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    num_epochs: int,
    checkpoint_path: str = "checkpoints/latest.pt",
    save_every: int = 1,
):

    logger = setup_logger()

    model.to(device)
    model.train()

    for epoch in range(num_epochs):

        total_loss = 0.0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Average Loss: {avg_loss:.4f}"
        )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                path=checkpoint_path,
            )


def main():

    config = GPTConfig(
        vocab_size=50257,
        context_length=128,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        qkv_bias=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(
        "resources/data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    tokens = tokenizer.encode(text)

    dataloader = create_dataloader(
        tokens=tokens,
        context_length=config.context_length,
        batch_size=32,
        shuffle=True,
    )

    model = GPTModel(config)

    criterion = GPTLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=3,
    )


if __name__ == "__main__":
    main()