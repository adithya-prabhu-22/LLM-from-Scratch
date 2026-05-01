import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model,
    optimizer,
    path,
    device="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded from {path}")

    return model, optimizer, epoch, loss