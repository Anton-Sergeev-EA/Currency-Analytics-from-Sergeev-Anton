import pandas as pd
from typing import Optional
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.data.cache import CacheManager


class DataService:
    def __init__(self):
        self.loader = DataLoader()
        self.cache = CacheManager()

    def get_historical_data(self, period_days: int = 90, refresh: bool = False) -> pd.DataFrame:
        if refresh:
            self.cache.delete(f"data_{period_days}")
        return self.loader.load_data(period_days)

    def get_current_rates(self) -> dict:
        df = self.loader.load_data(1)
        if len(df) == 0:
            return {"usd": 0.0, "eur": 0.0}
        return {
            "usd": float(df['usd_rate'].iloc[-1]),
            "eur": float(df['eur_rate'].iloc[-1])
        }
    