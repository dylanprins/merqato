from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.model import PriceTransformer
from data.dataset import create_dataloaders


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test set, gather predictions, targets (Actual), and weeks.
    """
    model.eval()
    predictions, targets, weeks = [], [], []
    with torch.no_grad():
        for x, y, _, w in dataloader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            predictions.extend(preds)
            targets.extend(y.numpy())
            weeks.extend(w.numpy())
    return np.array(predictions), np.array(targets), np.array(weeks)


def plot_results(preds, targets, weeks):
    """
        Plot actual vs predicted prices of strawberries with a 2-week horizon.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label='Actual', marker='o', alpha=0.7)
    plt.plot(preds, label='Predicted', marker='x', alpha=0.7)
    plt.xticks(ticks=np.arange(len(weeks)), labels=weeks, rotation=45)
    plt.title("Actual vs Predicted Prices of Strawberries with a 2-week horizon")
    plt.xlabel("Week")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def main(args):
    df = pd.read_parquet(args.data)

    _, _, test_loader = create_dataloaders(
        df,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        window_size=args.window_size,
        target_horizon=args.target_horizon,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PriceTransformer(
        input_dim=len(['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation']),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    preds, targets, weeks = evaluate_model(model, test_loader, device)

    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - preds) / targets)) * 100
    r2 = r2_score(targets, preds)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R^2: {r2:.4f}")

    plot_results(preds, targets, weeks)


if __name__ == '__main__':
    # We don't need the argument parser here, we should have saved the config of the model during training in a yaml file and load it
    # because the model is trained with a specific config, and we need to use the same config for inference. For now, I just saved the weights.
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='local_data/processed/data_2013_2023.parquet')
    parser.add_argument('--model_path', type=str, default='local_data/models/original.pt')
    parser.add_argument('--train_years', type=int, nargs='+', default=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument('--val_years', type=int, nargs='+', default=[2021])
    parser.add_argument('--test_years', type=int, nargs='+', default=[2022, 2023])
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--target_horizon', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
