import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_week_date_range(year, week):
    """
    Get the start and end date of a week given the year and week number.
    """
    start = date.fromisocalendar(year, week, 1)
    end = start + timedelta(days=6)
    return start, end


def time_aware_imputation(df, features, target, model, position):
    """
    Perform time-aware imputation for missing values in the target column using a regression model.
    In time series data, it's crucial to use only past data for training to avoid data leakage.
    """

    df_result = df.copy()
    pbar = tqdm(df_result[df_result[target].isna()].index, desc=f"Imputing {target}", position=position, leave=True)
    for idx in pbar:
        train_data = df_result.loc[:idx - 1]
        train_data = train_data[train_data[target].notna()]

        if train_data.empty:
            continue  # Skip if there's no past data to train on

        x_train = train_data[features]
        y_train = train_data[target]
        x_predict = df_result.loc[idx, features].to_frame().T

        model.fit(x_train, y_train)
        prediction = model.predict(x_predict)
        df_result.at[idx, target] = prediction[0]

    return df_result[target]


def impute_target_with_progress(args):
    df_sorted, features, target, pipeline, position = args
    model_pipeline = clone(pipeline)
    imputed = time_aware_imputation(df_sorted, features, target, model_pipeline, position)
    return target, imputed


def regression(df):
    """
    Perform regression to impute missing values in the target columns using a Random Forest model.

    Again, this is not recommended, but could work with larger datasets missing less values
    """
    features = ['year', 'week', 'windspeed', 'temp', 'cloudcover', 'precip', 'solarradiation']
    targets = ['price_min', 'price_max', 'price']

    # Normalize the features
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), features)
    ])

    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    df = df.sort_values(by=["year", "week"]).reset_index(drop=True)

    tasks = [(df, features, target, base_pipeline, i) for i, target in enumerate(targets)]

    with ProcessPoolExecutor(max_workers=len(targets)) as executor:
        results = executor.map(impute_target_with_progress, tasks)

    for target, imputed_series in results:
        df[target] = imputed_series

    return df


def preprocess(df, impute=False):
    # First column contains indexing starting at 1, we don't need it
    data = df.iloc[:, 1:]

    # Apply the lookup function to create new start and end dates, they were partially missing
    data['start_date'], data['end_date'] = zip(*data.apply(lambda row: get_week_date_range(row['year'], row['week']), axis=1))

    # This makes sense since there is only one category and unit in this dataset
    data['category'] = data['category'].fillna('I')
    data['unit'] = data['unit'].fillna('Euros/kg')

    # Fill missing price values using regression/imputation, using only real values is highly recommended though!
    if impute:
        # Do not impute the test & eval data, only the training data
        test_eval_data = data[data['year'].isin(args.test_eval_years)]
        data = regression(data[~data['year'].isin(args.test_eval_years)])
        data = pd.concat([data, test_eval_data], ignore_index=True)

    data.dropna(subset=['price_min', 'price_max', 'price'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    start_year = data['start_date'].min().year
    end_year = data['start_date'].max().year

    # Save the preprocessed data to memory efficient format
    os.makedirs('local_data/processed', exist_ok=True)
    data.to_parquet(f'local_data/processed/data_{start_year}_{end_year}{"_imputed" if impute else ""}.parquet', index=False)

    return data


if __name__ == '__main__':
    # Add argument parser to handle command line arguments, easily expandable
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='local_data/raw/senior_ds_test.csv', help="Path to the input data file")
    parser.add_argument("--impute", action="store_true", help="Impute for missing values, not recommended to maintain data integrity")
    parser.add_argument("--test_eval_years", type=int, nargs='+', default=[2021, 2022, 2023], help="Years to evaluate and test on")
    args = parser.parse_args()

    # Load data from the specified path
    df = pd.read_csv(args.data)
    data = preprocess(df, impute=args.impute)