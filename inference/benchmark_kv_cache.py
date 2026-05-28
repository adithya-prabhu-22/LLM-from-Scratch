import argparse
import csv
import json
import time
from dataclasses import fields
from pathlib import Path

import torch

from config.config import GPTConfig
from model.gpt_model import GPTModel


def load_config(config_path: str) -> GPTConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    valid_fields = {field.name for field in fields(GPTConfig)}
    filtered_config = {
        key: value
        for key, value in config_data.items()
        if key in valid_fields
    }

    return GPTConfig(**filtered_config)


def load_checkpoint(model: GPTModel, checkpoint_path: str, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    return model


def get_peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0

    return torch.cuda.max_memory_allocated(device) / 1024**2


@torch.no_grad()
def generate_without_kv_cache(
    model,
    input_ids,
    max_new_tokens: int,
):
    generated = input_ids

    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


@torch.no_grad()
def generate_with_kv_cache(
    model,
    input_ids,
    max_new_tokens: int,
):
    logits, past_kv = model(
        input_ids,
        use_cache=True,
    )

    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True,
    )

    generated = torch.cat(
        [input_ids, next_token],
        dim=1,
    )

    for _ in range(max_new_tokens - 1):
        logits, past_kv = model(
            next_token,
            past_kv=past_kv,
            use_cache=True,
        )

        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


def benchmark_once(
    model,
    device,
    vocab_size: int,
    prompt_length: int,
    max_new_tokens: int,
    use_kv_cache: bool,
):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, prompt_length),
        dtype=torch.long,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    if use_kv_cache:
        _ = generate_with_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    else:
        _ = generate_without_kv_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    latency = end_time - start_time
    tokens_per_sec = max_new_tokens / max(latency, 1e-8)
    peak_memory_mb = get_peak_memory_mb(device)

    return latency, tokens_per_sec, peak_memory_mb


def benchmark(
    model,
    config,
    device,
    prompt_lengths,
    max_new_tokens,
    warmup_runs,
    benchmark_runs,
):
    results = []

    model.eval()

    for prompt_length in prompt_lengths:
        total_length = prompt_length + max_new_tokens

        if total_length > config.context_length:
            print(
                f"Skipping prompt_length={prompt_length}, "
                f"max_new_tokens={max_new_tokens}, "
                f"because total length {total_length} exceeds "
                f"context_length {config.context_length}"
            )
            continue

        for _ in range(warmup_runs):
            benchmark_once(
                model=model,
                device=device,
                vocab_size=config.vocab_size,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
                use_kv_cache=False,
            )

            benchmark_once(
                model=model,
                device=device,
                vocab_size=config.vocab_size,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
                use_kv_cache=True,
            )

        no_cache_latencies = []
        no_cache_tokens_per_sec = []
        no_cache_memory = []

        kv_cache_latencies = []
        kv_cache_tokens_per_sec = []
        kv_cache_memory = []

        for _ in range(benchmark_runs):
            latency, tokens_per_sec, memory = benchmark_once(
                model=model,
                device=device,
                vocab_size=config.vocab_size,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
                use_kv_cache=False,
            )

            no_cache_latencies.append(latency)
            no_cache_tokens_per_sec.append(tokens_per_sec)
            no_cache_memory.append(memory)

            latency, tokens_per_sec, memory = benchmark_once(
                model=model,
                device=device,
                vocab_size=config.vocab_size,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
                use_kv_cache=True,
            )

            kv_cache_latencies.append(latency)
            kv_cache_tokens_per_sec.append(tokens_per_sec)
            kv_cache_memory.append(memory)

        avg_no_cache_latency = sum(no_cache_latencies) / len(no_cache_latencies)
        avg_kv_cache_latency = sum(kv_cache_latencies) / len(kv_cache_latencies)

        avg_no_cache_tps = sum(no_cache_tokens_per_sec) / len(no_cache_tokens_per_sec)
        avg_kv_cache_tps = sum(kv_cache_tokens_per_sec) / len(kv_cache_tokens_per_sec)

        avg_no_cache_memory = sum(no_cache_memory) / len(no_cache_memory)
        avg_kv_cache_memory = sum(kv_cache_memory) / len(kv_cache_memory)

        speedup = avg_no_cache_latency / max(avg_kv_cache_latency, 1e-8)

        row = {
            "prompt_length": prompt_length,
            "max_new_tokens": max_new_tokens,
            "latency_without_kv_sec": round(avg_no_cache_latency, 6),
            "latency_with_kv_sec": round(avg_kv_cache_latency, 6),
            "speedup": round(speedup, 4),
            "tokens_per_sec_without_kv": round(avg_no_cache_tps, 4),
            "tokens_per_sec_with_kv": round(avg_kv_cache_tps, 4),
            "peak_memory_without_kv_mb": round(avg_no_cache_memory, 2),
            "peak_memory_with_kv_mb": round(avg_kv_cache_memory, 2),
        }

        results.append(row)

        print(row)

    return results


def save_results(results, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt_length",
        "max_new_tokens",
        "latency_without_kv_sec",
        "latency_with_kv_sec",
        "speedup",
        "tokens_per_sec_without_kv",
        "tokens_per_sec_with_kv",
        "peak_memory_without_kv_mb",
        "peak_memory_with_kv_mb",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def parse_prompt_lengths(value: str):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPT inference with and without KV cache."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config.json",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output benchmark CSV file",
    )

    parser.add_argument(
        "--prompt_lengths",
        type=str,
        default="32,64,128",
        help="Comma-separated prompt lengths",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Number of new tokens to generate",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=2,
        help="Number of warmup runs",
    )

    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=5,
        help="Number of measured runs",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = load_config(args.config)

    model = GPTModel(config)
    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    prompt_lengths = parse_prompt_lengths(args.prompt_lengths)

    results = benchmark(
        model=model,
        config=config,
        device=device,
        prompt_lengths=prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )

    save_results(
        results=results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()