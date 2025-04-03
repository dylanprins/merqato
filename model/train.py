from argparse import ArgumentParser

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from model.model import PriceTransformer
from data.dataset import create_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch. Simple training loop measuring loss and MAE.
    """
    model.train()
    running_loss = 0.0
    total_absolute_error = 0.0
    for x, y, _, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        total_absolute_error += torch.sum(torch.abs(outputs - y)).item()
    avg_loss = running_loss / len(dataloader.dataset)
    avg_mae = total_absolute_error / len(dataloader.dataset)
    return avg_loss, avg_mae


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation or test set.
    """
    model.eval()
    running_loss = 0.0
    total_absolute_error = 0.0
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            total_absolute_error += torch.sum(torch.abs(outputs - y)).item()
    avg_loss = running_loss / len(dataloader.dataset)
    avg_mae = total_absolute_error / len(dataloader.dataset)
    return avg_loss, avg_mae


def main(args):
    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project, config=vars(args)) # Comment this and the .log()'s if you don't want to use wandb

    df = pd.read_parquet(args.data)
    train_loader, val_loader, test_loader = create_dataloaders(
        df,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        window_size=args.window_size,
        target_horizon=args.target_horizon,
        batch_size=args.batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize simple transformer model
    model = PriceTransformer(
        input_dim=len(['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation']),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae
        })

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"local_data/models/{wandb.run.id}.pt")

    print("Training complete. Best Val Loss:", best_val_loss)
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print("Test Loss:", test_loss, "Test MAE:", test_mae)
    wandb.log({"test_loss": test_loss, "test_mae": test_mae})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='local_data/processed/data_2013_2023.parquet', help="Path to the input (preprocessed) data file")
    parser.add_argument('--wandb_project', type=str, default="merqato")
    parser.add_argument('--train_years', type=int, nargs='+', default=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument('--val_years', type=int, nargs='+', default=[2021], help="Years to use for validation")
    parser.add_argument('--test_years', type=int, nargs='+', default=[2022, 2023], help="Years to use for testing")
    parser.add_argument('--window_size', type=int, default=12, help="Sequence length of the input data")
    parser.add_argument('--target_horizon', type=int, default=2, help="Prediction horizon")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
