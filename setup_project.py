import os

structure = {
    
    "config": ["config.py"],
    
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
    ],
    
    "notebooks": [],
    "resources": [],
}

root_files = [
    "main.py",
    "requirements.txt",
    "README.md",
]

def create_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"# {os.path.basename(path)}\n")

def create_structure():
    
    for folder, files in structure.items():
        os.makedirs(folder, exist_ok=True)

        
        if folder not in ["notebooks", "resources"]:
            create_file(os.path.join(folder, "__init__.py"))

        for file in files:
            create_file(os.path.join(folder, file))

    
    for file in root_files:
        create_file(file)

    print("✅ Project structure created in current repo root!")

if __name__ == "__main__":
    create_structure()