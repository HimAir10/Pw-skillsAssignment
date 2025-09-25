import pandas as pd

def load_cryptocurrency_data(file_paths):
    """Load and concatenate multiple CSV files"""
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def clean_data(df):
    """Clean and preprocess the dataset"""
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.drop_duplicates(inplace=True)
    return df
