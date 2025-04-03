# Merqato Price Forecasting

A time-series forecasting project designed to predict strawberry prices with a 2-week horizon using a simple Transformer model.
The dataset includes weekly data between 2013 and 2023 with weather and pricing features.

## 📁 Project Structure

```
├── data/
│   ├── dataset.py         # PyTorch Dataset and DataLoader split logic
│   └── preprocess.py      # Preprocessing, time-aware imputation logic
│
├── local_data/
│   ├── models/            # Trained model weights (e.g. model.pt)
│   ├── plots/             # Saved performance plots
│   ├── processed/         # Preprocessed parquet files
│   └── raw/               # Raw CSV data
│
├── model/
│   ├── model.py           # Transformer model definition
│   ├── simulate.py        # Evaluation script with metrics + matplotlib plots
│   └── train.py           # Training loop with wandb integration
```

## 🧠 Core Functionality

### 1. Preprocessing
- Safe handling of missing values
- Weekly date reconstruction
- Time-aware imputation of prices with Random Forest regressors if desired

```bash
python -m data.preprocess --data local_data/raw/senior_ds_test.csv --impute
```

### 2. Dataset
- Custom `PriceForecastDataset` for rolling-window normalization
- Splits data by year into train/val/test
- Efficient `DataLoader` setup

### 3. Model
- Minimal Transformer encoder
- Sequence-to-one prediction
- Continuous price output

### 4. Training
- Logs to Weights & Biases (wandb)
- Early saving of best model

```bash
python -m model.train --data local_data/processed/data_2013_2023.parquet
```

### 5. Evaluation
- MAE, RMSE, MAPE, R² metrics
- Matplotlib plots for predictions vs actuals

```bash
python -m model.simulate --model_path local_data/models/original.pt
```

## ⚙️ Requirements
Install everything with:
```bash
conda env create -f environment.yml
```

Then install preferred PyTorch in that environment, e.g.:
```bash
conda activate dylan-merqato

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 📌 Notes
- All preprocessing is done **without data leakage** (only past data is used).
- The model is very lightweight and intended as a prototype — you can easily extend it with more features, attention masks, or richer architectures.
- Data is stored locally to prevent large files in version control.

## 👋 Author
Dylan Prins @ Neurality for Merqato · 2025

