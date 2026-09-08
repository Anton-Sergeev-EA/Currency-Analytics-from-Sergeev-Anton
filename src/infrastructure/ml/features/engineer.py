import pandas as pd


class FeatureEngineer:
    """
    Класс для создания признаков (lags, rolling stats, calendar features) из временных рядов.
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Календарные признаки
            df["day_of_week"] = df["date"].dt.dayofweek
            df["day"] = df["date"].dt.day
            df["month"] = df["date"].dt.month

        # Лаги и скользящие показатели для валютных пар
        for col in ["usd_rate", "eur_rate"]:
            if col in df.columns:
                for lag in [1, 2, 3, 7]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                for window in [3, 7]:
                    df[f"{col}_rolling_mean_{window}"] = df[col].shift(1).rolling(window=window).mean()
                    df[f"{col}_rolling_std_{window}"] = df[col].shift(1).rolling(window=window).std()

        return df