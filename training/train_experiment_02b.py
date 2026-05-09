import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config.config import GPTConfig
from model.gpt_model import GPTModel
from training.loss import GPTLoss
from training.evaluate import evaluate
from data.dataloader import create_dataloader
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger


def setup_custom_bpe_tokenizer():
    possible_paths = [
        Path("/content/Domain-Specific-BPE-Tokenizer"),
        Path(r"C:\Users\ADITHYA\Desktop\Domain-Specific-BPE-Tokenizer"),
    ]

    tokenizer_repo = None

    for path in possible_paths:
        if path.exists():
            tokenizer_repo = path
            break

    if tokenizer_repo is None:
        raise FileNotFoundError(
            "Domain-Specific-BPE-Tokenizer repo not found. "
            "Clone it in Colab or place it on Desktop in Windows."
        )

    tokenizer_data_src = tokenizer_repo / "data"
    tokenizer_data_alias = tokenizer_repo / "tokenizer_data"

    if not tokenizer_data_alias.exists():
        shutil.copytree(tokenizer_data_src, tokenizer_data_alias)

        for py_file in tokenizer_data_alias.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            text = text.replace("from data.", "from tokenizer_data.")
            py_file.write_text(text, encoding="utf-8")

    sys.path.insert(0, str(tokenizer_repo))

    from tokenizer_data.bpe_tokenizer import BPETokenizer

    return BPETokenizer


BPETokenizer = setup_custom_bpe_tokenizer()


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


def save_tokenizer_report(tokens, text, tokenizer_name, experiment_dir):
    total_chars = len(text)
    total_words = len(text.split())
    total_tokens = len(tokens)

    report = {
        "tokenizer": tokenizer_name,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "chars_per_token": total_chars / total_tokens,
        "tokens_per_word": total_tokens / total_words,
        "words_per_token": total_words / total_tokens,
    }

    report_path = Path(experiment_dir) / "tokenizer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report


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
    num_epochs = 5
    batch_size = 8

    save_config(config, experiment_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("resources/data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer.load("resources/bpe_medical_52k.json")
    tokens = tokenizer.encode(text)

    save_tokenizer_report(
        tokens=tokens,
        text=text,
        tokenizer_name="custom_bpe_52k_medical",
        experiment_dir=experiment_dir,
    )

    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_dataloader = create_dataloader(
        tokens=train_tokens,
        context_length=config.context_length,
        batch_size=batch_size,
        shuffle=True,
        stride=config.stride,
    )

    val_dataloader = create_dataloader(
        tokens=val_tokens,
        context_length=config.context_length,
        batch_size=batch_size,
        shuffle=False,
        stride=config.stride,
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