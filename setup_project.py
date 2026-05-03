import os
import sys


structure = {
    "config": [
        "config.py",
    ],

    "data": [
        "tokenizer.py",
        "bpe_tokenizer.py",
        "dataloader.py",
        "input_target_pairs.py",
    ],

    "model": [
        "embeddings.py",
        "layer_norm.py",
        "activations.py",
        "feedforward.py",
        "transformer_block.py",
        "gpt_model.py",
    ],

    "model/attention": [
        "self_attention.py",
        "causal_attention.py",
        "multihead_attention.py",
    ],

    "training": [
        "loss.py",
        "train.py",
        "evaluate.py",
    ],

    "inference": [
        "generate.py",
    ],

    "utils": [
        "checkpoint.py",
        "logger.py",
    ],

    "tests": [
        "test_model.py",
        "test_attention.py",
        "test_data.py",
    ],

    "experiments/experiment_01":       [],
    "experiments/experiment_01/plots": [],

    "notebooks": [],

    "resources": [],

    "scripts": [
        "train_run.sh",
        "prepare_medical_5m.py",
    ],

    "checkpoints": [],

    "logs": [],

    "assets": [],
}


root_files = [
    "main.py",
    "README.md",
    "setup_project.py",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".env.example",
]


NON_PACKAGE_DIRS = {
    "notebooks",
    "resources",
    "experiments",
    "experiments/experiment_01",
    "experiments/experiment_01/plots",
    "scripts",
    "checkpoints",
    "logs",
    "assets",
}


GITIGNORE_CONTENT = """\
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/
ENV/

# Checkpoints & model weights
checkpoints/
*.pt
*.pth
*.bin
*.safetensors

# Logs
logs/
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment variables
.env

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
"""


ENV_EXAMPLE_CONTENT = """\
PROJECT_NAME=gpt-experiment
DATA_PATH=resources/data.txt
CHECKPOINT_PATH=checkpoints/latest.pt
LOG_PATH=logs/
"""


REQUIREMENTS_CONTENT = """\
torch>=2.2.0
numpy>=1.26.0
tiktoken>=0.6.0
regex>=2023.0.0
requests>=2.31.0
tqdm>=4.66.0
datasets>=2.19.0
transformers>=4.40.0
accelerate>=0.29.0
sentencepiece>=0.2.0
"""


REQUIREMENTS_DEV_CONTENT = """\
-r requirements.txt

pytest>=8.0.0
black>=24.0.0
ruff>=0.4.0
ipykernel>=6.29.0
matplotlib>=3.8.0
networkx>=3.2.0
sympy>=1.12.0
jupyter>=1.0.0
tensorboard>=2.15.0
seaborn>=0.13.0
pandas>=2.2.0
"""


TRAIN_SCRIPT_CONTENT = """\
#!/bin/bash

python -m training.train
"""


CONTENT_MAP = {
    ".gitignore": GITIGNORE_CONTENT,
    ".env.example": ENV_EXAMPLE_CONTENT,
    "requirements.txt": REQUIREMENTS_CONTENT,
    "requirements-dev.txt": REQUIREMENTS_DEV_CONTENT,
    "train_run.sh": TRAIN_SCRIPT_CONTENT,
}


stats = {
    "created": 0,
    "skipped": 0,
    "dirs_created": 0,
    "dirs_existed": 0,
}


def safe_mkdir(path: str) -> None:

    if not os.path.exists(path):
        os.makedirs(
            path,
            exist_ok=True,
        )

        print(f"  [dir created]  {path}/")

        stats["dirs_created"] += 1

    else:
        print(f"  [dir exists]   {path}/")

        stats["dirs_existed"] += 1


def safe_write(
    path: str,
    content: str = "",
) -> None:

    if os.path.exists(path):
        print(
            f"    [skip]     {path} "
            "(already exists — untouched)"
        )

        stats["skipped"] += 1

        return

    try:
        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

        print(f"    [created]  {path}")

        stats["created"] += 1

    except OSError as error:
        print(
            f"    [ERROR]    Could not create {path}: {error}",
            file=sys.stderr,
        )


def create_structure() -> None:

    print("\n=== Setting up project structure ===")

    print(
        "NOTE: Existing files "
        "will NEVER be overwritten.\n"
    )

    for folder, files in structure.items():

        safe_mkdir(folder)

        if folder not in NON_PACKAGE_DIRS:

            init_path = os.path.join(
                folder,
                "__init__.py",
            )

            safe_write(init_path)

        for filename in files:

            file_path = os.path.join(
                folder,
                filename,
            )

            content = CONTENT_MAP.get(
                filename,
                "",
            )

            safe_write(
                file_path,
                content,
            )

    print("\n[root files]")

    for filename in root_files:

        content = CONTENT_MAP.get(
            filename,
            "",
        )

        safe_write(
            filename,
            content,
        )

    print("\n" + "=" * 50)

    print("Done.")

    print(
        f"  Directories : "
        f"{stats['dirs_created']} created, "
        f"{stats['dirs_existed']} already existed"
    )

    print(
        f"  Files       : "
        f"{stats['created']} created, "
        f"{stats['skipped']} skipped"
    )

    print("=" * 50)

    print_tree()


def print_tree() -> None:

    print("\nProject layout:")

    print(".")

    for folder in structure:

        print(f"├── {folder}/")

        files = structure[folder]

        all_files = []

        if folder not in NON_PACKAGE_DIRS:
            all_files.append("__init__.py")

        all_files.extend(files)

        for index, filename in enumerate(all_files):

            connector = (
                "└──"
                if index == len(all_files) - 1
                else "├──"
            )

            print(f"│   {connector} {filename}")

    print("├── (root)")

    for index, filename in enumerate(root_files):

        connector = (
            "└──"
            if index == len(root_files) - 1
            else "├──"
        )

        print(f"    {connector} {filename}")


if __name__ == "__main__":

    cwd = os.getcwd()

    print(f"\nRunning from: {cwd}")

    confirm = input(
        "Create project structure here? [y/N]: "
    ).strip().lower()

    if confirm != "y":

        print("Aborted. No files were created.")

        sys.exit(0)

    create_structure()