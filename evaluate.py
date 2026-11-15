import os
import torch
import matplotlib.pyplot as plt

def plot_curves(history, out_dir):
    # NEW: 2 graphs. Left = loss (lower is better). Right = val Dice (higher is better).
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(epochs, history["train_loss"], label="train loss")
    ax[0].plot(epochs, history["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["val_dice"], color="green", label="val Dice")
    ax[1].set_xlabel("epoch")
    ax[1].set_title("Validation Dice")
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curves.png"))
    plt.close()
    print(f"Saved training graphs in {out_dir}/curves.png")


def show_predictions(model, loader, device, out_dir, name="val", n=5):
    # Save n pictures: X-ray | true mask | predicted mask | both on top of each other
    model.eval()
    batch = next(iter(loader))
    images = batch["xray"].to(device)

    with torch.no_grad():
        preds = (torch.sigmoid(model(images)) > 0.5).float()

    for i in range(min(n, len(images))):
        xray = images[i, 0].cpu()
        true = batch["mask"][i, 0]
        pred = preds[i, 0].cpu()

        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].imshow(xray, cmap="gray")
        ax[0].set_title("X-ray")
        ax[1].imshow(true, cmap="gray")
        ax[1].set_title("True mask")
        ax[2].imshow(pred, cmap="gray")
        ax[2].set_title("Prediction")
        # NEW: overlay. Green = true mask outline, red = prediction outline
        ax[3].imshow(xray, cmap="gray")
        ax[3].contour(true, levels=[0.5], colors="green")
        ax[3].contour(pred, levels=[0.5], colors="red")
        ax[3].set_title("Green = true, Red = pred")
        for a in ax:
            a.axis("off")
        plt.savefig(os.path.join(out_dir, f"{name}_prediction_{i}.png"))
        plt.close()

    print(f"Saved {min(n, len(images))} {name} prediction pictures in {out_dir}")
