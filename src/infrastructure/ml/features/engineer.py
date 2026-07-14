import pandas as pd
import numpy as np


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 10:
        return df

    df = df.copy()

    for col in ['usd_rate', 'eur_rate']:
        for lag in [1, 3, 7, 14]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for col in ['usd_rate', 'eur_rate']:
        for window in [7, 14]:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()

    df['usd_eur_diff'] = df['usd_rate'] - df['eur_rate']
    df['usd_eur_ratio'] = df['usd_rate'] / df['eur_rate']
    df['usd_eur_corr_14'] = df['usd_rate'].rolling(14).corr(df['eur_rate'])

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df = df.dropna()

    return df
