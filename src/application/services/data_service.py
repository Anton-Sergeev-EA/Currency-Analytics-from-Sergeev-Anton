from typing import Dict, Any
from src.infrastructure.data.loader import DataLoader
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class DataService:
    """Сервис подготовки данных и аналитической статистики."""

    def __init__(self):
        self.loader = DataLoader()

    async def get_historical_data(self, period_days: int = 180, refresh: bool = False) -> Dict[str, Any]:
        """
        Возвращает исторические данные в универсальном формате:
        как в виде списков для графиков (dates, usd, eur), так и в виде списка записей (records).

        `refresh=True` bypasses the loader's cache and re-fetches from the
        CBR API (used by the admin /api/refresh and /api/force-refresh
        endpoints) - previously this parameter didn't exist at all, so
        calling it always silently served cached data and, when passed as
        a keyword argument by the admin refresh endpoint, raised
        TypeError instead.
        """
        df = await (self.loader.refresh_data(period_days) if refresh else self.loader.load_data(period_days))
        if df is None or df.empty:
            return {"dates": [], "usd": [], "eur": [], "usd_rate": [], "eur_rate": [], "records": []}

        df_copy = df.copy()
        if "date" in df_copy.columns:
            df_copy["date_str"] = df_copy["date"].dt.strftime("%Y-%m-%d")
        else:
            df_copy["date_str"] = []

        dates = df_copy["date_str"].tolist()
        usd = [round(float(x), 2) for x in df_copy["usd_rate"]] if "usd_rate" in df_copy.columns else []
        eur = [round(float(x), 2) for x in df_copy["eur_rate"]] if "eur_rate" in df_copy.columns else []

        records = df_copy.to_dict(orient="records")

        return {
            "dates": dates,
            "usd": usd,
            "eur": eur,
            "usd_rate": usd,
            "eur_rate": eur,
            "records": records,
            "data": records
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Расчет текущих показателей курсов и изменений за день."""
        df = await self.loader.load_data(365)
        if df is None or df.empty:
            return {
                "usd_current": 81.41,
                "usd_change": 0.20,
                "eur_current": 94.06,
                "eur_change": -0.15,
                "total_records": 0
            }

        usd_curr = float(df["usd_rate"].iloc[-1]) if "usd_rate" in df.columns else 81.41
        eur_curr = float(df["eur_rate"].iloc[-1]) if "eur_rate" in df.columns else 94.06

        usd_prev = float(df["usd_rate"].iloc[-2]) if len(df) > 1 and "usd_rate" in df.columns else usd_curr
        eur_prev = float(df["eur_rate"].iloc[-2]) if len(df) > 1 and "eur_rate" in df.columns else eur_curr

        return {
            "usd_current": round(usd_curr, 2),
            "usd_change": round(usd_curr - usd_prev, 2),
            "eur_current": round(eur_curr, 2),
            "eur_change": round(eur_curr - eur_prev, 2),
            "total_records": len(df)
        }
