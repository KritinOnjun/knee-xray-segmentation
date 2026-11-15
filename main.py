from args import get_args
import pandas as pd
import os
from datasets import KneeDataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model, validate_model
from evaluate import show_predictions
import torch
import torch.nn as nn

if __name__ == "__main__":
    # 1: we need arguments
    args = get_args()

    # 2: Loading the CSV files
    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))  # FIX: csv_dir, not data_dir
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # 3: Define datasets (augmentation only for training)
    train_dataset = KneeDataset(train_set, augment=True)
    val_dataset = KneeDataset(val_set, augment=False)

    # 4: Create data dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5: Initialize the model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        depth=5,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # 6: Train (FIX: training was never called before)
    best_path = train_model(model, train_loader, val_loader, device, args)

    # 7: Load the best model and look at some predictions
    model.load_state_dict(torch.load(best_path, map_location=device))
    show_predictions(model, val_loader, device, args.out_dir, name="val")

    # 8: final score on the TEST set
    test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))
    test_loader = DataLoader(KneeDataset(test_set), batch_size=args.batch_size, shuffle=False)
    test_loss, test_dice = validate_model(model, test_loader, nn.BCEWithLogitsLoss(), device)
    print(f"TEST: loss {test_loss:.4f}  Dice {test_dice:.4f}  ({len(test_set)} pictures)")
    show_predictions(model, test_loader, device, args.out_dir, name="test")
