import torch
import tiktoken

from config.config import GPTConfig
from model.gpt_model import GPTModel


def generate(
    model,
    input_ids,
    max_new_tokens: int,
    context_length: int,
):

    model.eval()

    for _ in range(max_new_tokens):

        input_ids_cond = input_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_ids_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.argmax(
            probs,
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return input_ids


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

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(config).to(device)

    checkpoint = torch.load(
        "checkpoints/latest.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    prompt = "Deep learning"

    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=50,
        context_length=config.context_length,
    )

    output_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print(output_text)


if __name__ == "__main__":
    main()