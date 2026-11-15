import os
import torch
import torch.nn as nn
from utils import dice_loss_from_logits, dice_score_from_logits
from evaluate import plot_curves

def train_model(model, train_loader, val_loader, device, args):
    model = model.to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_model.pth")
    best_loss = float("inf")  # start "infinitely bad", so epoch 1 is always better

    # NEW: remember the numbers of every epoch, to draw graphs at the end
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch["xray"].to(device)
            masks = data_batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = validate_model(model, val_loader, bce, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch + 1}/{args.epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  val Dice {val_dice:.4f}")

        # CHANGED: keep the model with the LOWEST val loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print("  -> lowest val loss so far, saved")

    print(f"Best (lowest) val loss: {best_loss:.4f}")
    plot_curves(history, args.out_dir)  # NEW: save the graphs
    return best_path


def validate_model(model, val_loader, bce, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch["xray"].to(device)
            masks = data_batch["mask"].to(device)
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            running_dice += dice_score_from_logits(outputs, masks)

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)
    return val_loss, val_dice
