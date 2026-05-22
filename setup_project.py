from pathlib import Path


PROJECT_DIRS = [
    "config",
    "data",
    "model",
    "model/attention",
    "training",
    "inference",
    "utils",
    "tests",
    "experiments",
    "experiments/experiment_01",
    "experiments/experiment_01/plots",
    "experiments/experiment_02A",
    "experiments/experiment_02A/plots",
    "experiments/experiment_02B",
    "experiments/experiment_02B/plots",
    "experiments/experiment_03A",
    "experiments/experiment_03A/plots",
    "notebooks",
    "resources",
    "scripts",
    "checkpoints",
    "logs",
    "assets",
]


PROJECT_FILES = [
    "config/__init__.py",
    "config/config.py",

    "data/__init__.py",
    "data/tokenizer.py",
    "data/bpe_tokenizer.py",
    "data/dataloader.py",
    "data/input_target_pairs.py",

    "model/__init__.py",
    "model/embeddings.py",
    "model/layer_norm.py",
    "model/activations.py",
    "model/feedforward.py",
    "model/transformer_block.py",
    "model/gpt_model.py",

    "model/attention/__init__.py",
    "model/attention/self_attention.py",
    "model/attention/causal_attention.py",
    "model/attention/multihead_attention.py",

    "training/__init__.py",
    "training/loss.py",
    "training/train.py",
    "training/train_experiment_02b.py",
    "training/train_experiment_03a.py",
    "training/evaluate.py",
    "training/chunk_dataset.py",

    "inference/__init__.py",
    "inference/generate.py",
    "inference/benchmark_kv_cache.py",

    "utils/__init__.py",
    "utils/checkpoint.py",
    "utils/logger.py",

    "tests/__init__.py",
    "tests/test_model.py",
    "tests/test_attention.py",
    "tests/test_data.py",

    "scripts/train_run.sh",
    "scripts/prepare_data.py",
    "scripts/prepare_medical_5m.py",
    "scripts/tokenize_chunks_custom_bpe.py",
    "scripts/prepare_exp03a_30m_text_chunks.py",
    "scripts/tokenize_exp03a_custom_bpe.py",

    "experiments/experiment_01/README.md",
    "experiments/experiment_02A/README.md",
    "experiments/experiment_02B/README.md",
    "experiments/experiment_03A/README.md",

    "main.py",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".env.example",
]


def create_project_structure():
    root = Path.cwd()

    for directory in PROJECT_DIRS:
        (root / directory).mkdir(parents=True, exist_ok=True)

    for file_path in PROJECT_FILES:
        path = root / file_path

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


if __name__ == "__main__":
    create_project_structure()
