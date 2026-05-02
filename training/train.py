import torch
import tiktoken

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config.config import GPTConfig
from model.gpt_model import GPTModel
from training.loss import GPTLoss
from training.evaluate import evaluate
from data.dataloader import create_dataloader
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger


def train(
    config,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
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

        for batch_idx, (input_ids, targets) in enumerate(train_dataloader):

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)

            loss = criterion(logits, targets)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:

                current_lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_dataloader)}] "
                    f"Train Loss: {loss.item():.4f} "
                    f"LR: {current_lr:.8f}"
                )

        scheduler.step()

        train_loss = total_loss / len(train_dataloader)

        val_loss, perplexity = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Perplexity: {perplexity:.4f} "
            f"LR: {current_lr:.8f}"
        )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_loss,
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
        grad_clip=1.0,
        learning_rate=3e-4,
        min_learning_rate=1e-5,
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

    split_idx = int(0.9 * len(tokens))

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_dataloader = create_dataloader(
        tokens=train_tokens,
        context_length=config.context_length,
        batch_size=32,
        shuffle=True,
    )

    val_dataloader = create_dataloader(
        tokens=val_tokens,
        context_length=config.context_length,
        batch_size=32,
        shuffle=False,
    )

    model = GPTModel(config)

    criterion = GPTLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=3,
        eta_min=config.min_learning_rate,
    )

    train(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=3,
    )


if __name__ == "__main__":
    main()