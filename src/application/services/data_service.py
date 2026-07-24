import pandas as pd
from src.infrastructure.data.loader import DataLoader


class DataService:
    def __init__(self, loader: DataLoader = None):
        self.loader = loader or DataLoader()

    async def get_historical_data(self, period_days: int = 180, refresh: bool = False) -> pd.DataFrame:
        """
        Получает исторические данные котировок через DataLoader.
        """
        return await self.loader.load_data(period_days)
    
