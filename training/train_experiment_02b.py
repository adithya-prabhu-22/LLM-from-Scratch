import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

from config.config import GPTConfig
from model.gpt_model import GPTModel
from training.loss import GPTLoss
from training.evaluate import evaluate
from training.chunk_dataset import TokenChunkDataset
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def get_gpu_memory_stats(device):
    if device.type != "cuda":
        return 0.0, 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2

    return allocated, reserved, peak


def save_config(config, experiment_dir):
    config_path = Path(experiment_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=4)


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
    experiment_dir: str,
    save_every: int = 1,
):
    logger = setup_logger()

    experiment_dir = Path(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_logger = CSVLogger(
        experiment_dir / "train_log.csv",
        [
            "epoch",
            "batch_idx",
            "optimizer_step",
            "train_loss",
            "learning_rate",
            "tokens_seen",
            "tokens_per_sec",
            "gpu_allocated_mb",
            "gpu_reserved_mb",
            "gpu_peak_mb",
        ],
    )

    val_logger = CSVLogger(
        experiment_dir / "val_log.csv",
        [
            "epoch",
            "optimizer_step",
            "train_loss",
            "val_loss",
            "perplexity",
            "learning_rate",
        ],
    )

    model.to(device)
    model.train()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.use_amp and device.type == "cuda",
    )

    global_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        total_loss = 0.0
        accum_tokens = 0
        step_start_time = time.time()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()

        for batch_idx, (input_ids, targets) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            batch_tokens = input_ids.numel()
            accum_tokens += batch_tokens
            tokens_seen += batch_tokens

            with torch.amp.autocast(
                "cuda",
                enabled=config.use_amp and device.type == "cuda",
            ):
                logits = model(input_ids)
                loss = criterion(logits, targets)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            should_step = (batch_idx + 1) % config.gradient_accumulation_steps == 0
            is_last_batch = batch_idx == len(train_dataloader) - 1

            if should_step or is_last_batch:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                scheduler.step()
                global_step += 1

                step_time = time.time() - step_start_time
                tokens_per_sec = accum_tokens / max(step_time, 1e-8)

                gpu_allocated, gpu_reserved, gpu_peak = get_gpu_memory_stats(device)
                current_lr = optimizer.param_groups[0]["lr"]
                train_loss_value = loss.item() * config.gradient_accumulation_steps

                train_logger.log(
                    {
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx,
                        "optimizer_step": global_step,
                        "train_loss": round(train_loss_value, 6),
                        "learning_rate": current_lr,
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": round(tokens_per_sec, 2),
                        "gpu_allocated_mb": round(gpu_allocated, 2),
                        "gpu_reserved_mb": round(gpu_reserved, 2),
                        "gpu_peak_mb": round(gpu_peak, 2),
                    }
                )

                accum_tokens = 0
                step_start_time = time.time()

            total_loss += loss.item() * config.gradient_accumulation_steps

            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_dataloader)}] "
                    f"Train Loss: {loss.item() * config.gradient_accumulation_steps:.4f} "
                    f"LR: {current_lr:.8f}"
                )

        train_loss = total_loss / len(train_dataloader)

        val_loss, perplexity = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        val_logger.log(
            {
                "epoch": epoch + 1,
                "optimizer_step": global_step,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "perplexity": round(perplexity, 4),
                "learning_rate": current_lr,
            }
        )

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Perplexity: {perplexity:.4f} "
            f"LR: {current_lr:.8f}"
        )

        latest_path = checkpoint_dir / "latest.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=val_loss,
            path=str(latest_path),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_loss,
                path=str(best_path),
            )

            logger.info(
                f"Saved new best checkpoint at epoch {epoch + 1} "
                f"with val loss {val_loss:.4f}"
            )

        if (epoch + 1) % save_every == 0:
            epoch_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_loss,
                path=str(epoch_path),
            )

        model.train()


def main():
    config = GPTConfig(
        vocab_size=52000,
        context_length=256,
        stride=64,
        d_model=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        qkv_bias=False,
        grad_clip=1.0,
        learning_rate=3e-4,
        min_learning_rate=1e-5,
        use_amp=True,
        gradient_accumulation_steps=8,
    )

    experiment_dir = "experiments/exp_02b_custom_bpe_medical_5m"
    tokenized_dir = "/content/drive/MyDrive/exp_02b_custom_bpe/tokenized_chunks"

    num_epochs = 5
    batch_size = 8

    save_config(config, experiment_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TokenChunkDataset(
        tokenized_dir=tokenized_dir,
        context_length=config.context_length,
        stride=config.stride,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = GPTModel(config)
    criterion = GPTLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
    )

    total_steps = math.ceil(
        (num_epochs * len(train_dataloader))
        / config.gradient_accumulation_steps
    )

    warmup_steps = min(120, max(1, total_steps // 10))
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=config.min_learning_rate,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
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
        num_epochs=num_epochs,
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()