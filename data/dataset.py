import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class PriceForecastDataset(Dataset):
    """
    A PyTorch Dataset for time series forecasting of price data.

    I can improve this by not gathering all samples beforehand. But by indexing them in the __getitem__ method.
    This would be more efficient for large datasets, as it would avoid storing all samples in memory at once.
    Now it is fine because we have a small DF.

    To add this efficiency, we normalize the full df with a rolling window beforehand. Convert the df to a tensor and
    then index it in the __getitem__ method.
    """
    def __init__(self, df, window_size=12, target_horizon=2):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size # Now (norm) window size is equal to seq_len this is probably not the best choice
        self.target_horizon = target_horizon
        self.features = ['windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation']
        self.targets = self.df['price'].values # We can add functionality to predict price_min and price_max
        self.samples = []

        for idx in range(window_size, len(self.df) - target_horizon):
            past_window = self.df.loc[idx - window_size:idx - 1, self.features]
            future_price = self.targets[idx + target_horizon] # predict price over target_horizon weeks

            if np.isnan(future_price):
                continue  # skip if future target is missing

            # Simple normalization of window/sample
            mean = past_window.mean()
            std = past_window.std().replace(0, 1)  # avoid division by zero
            normalized_window = (past_window - mean) / std

            sample_year = self.df.loc[idx, 'year']
            sample_week = self.df.loc[idx + target_horizon, 'week']
            self.samples.append((normalized_window.values.astype(np.float32),
                                 np.float32(future_price), sample_year, sample_week))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features_seq, target, year, week = self.samples[idx]
        return torch.tensor(features_seq), torch.tensor(target), year, week


def split_dataset_indices(dataset, train_years, val_years, test_years):
    train_indices = [i for i, (_, _, year, _) in enumerate(dataset.samples) if year in train_years]
    val_indices = [i for i, (_, _, year, _) in enumerate(dataset.samples) if year in val_years]
    test_indices = [i for i, (_, _, year, _) in enumerate(dataset.samples) if year in test_years]
    return train_indices, val_indices, test_indices


def create_dataloaders(df, train_years, val_years, test_years, window_size=12, target_horizon=2, batch_size=32):
    dataset = PriceForecastDataset(df, window_size, target_horizon)
    train_idx, val_idx, test_idx = split_dataset_indices(dataset, train_years, val_years, test_years)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    df = pd.read_parquet("local_data/processed/data_2013_2023_imputed.parquet")
    train, val, test = create_dataloaders(df,
                                          train_years=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
                                          val_years=[2021],
                                          test_years=[2022, 2023],
                                          window_size=12,
                                          target_horizon=2,
                                          batch_size=32)

    for batch in train:
        features, target, year, week = batch
        print("Features shape:", features.shape)
        print("Target shape:", target.shape)
        print("Year shape:", year.shape)
        break