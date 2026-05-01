import math
import torch


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for input_ids, targets in dataloader:

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)

            loss = criterion(
                logits,
                targets,
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    perplexity = math.exp(avg_loss)

    model.train()

    return avg_loss, perplexity