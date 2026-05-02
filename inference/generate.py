import torch
import tiktoken

from config.config import GPTConfig
from model.gpt_model import GPTModel


def generate(
    model,
    input_ids,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):

    model.eval()

    for _ in range(max_new_tokens):

        input_ids_cond = input_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_ids_cond)

        logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )
        else:
            logits = logits / temperature

            if top_k is not None:
                top_values, _ = torch.topk(
                    logits,
                    k=top_k,
                    dim=-1,
                )

                min_top_value = top_values[:, -1].unsqueeze(-1)

                logits = torch.where(
                    logits < min_top_value,
                    torch.tensor(
                        float("-inf"),
                        device=logits.device,
                    ),
                    logits,
                )

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
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
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
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
        max_new_tokens=config.max_new_tokens,
        context_length=config.context_length,
        temperature=config.temperature,
        top_k=config.top_k,
    )

    output_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print(output_text)


if __name__ == "__main__":
    main()