import math
from typing import Dict, Any
import pandas as pd
from src.core.constants import SUPPORTED_CURRENCIES, short_code
from src.infrastructure.data.loader import DataLoader
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class DataService:
    """Сервис подготовки данных и аналитической статистики."""

    def __init__(self):
        self.loader = DataLoader()

    async def get_historical_data(self, period_days: int = 180, refresh: bool = False) -> Dict[str, Any]:
        """
        Возвращает исторические данные в универсальном формате: как в виде
        списков для графиков (dates, usd, eur, cny, gbp, ...), так и в виде
        списка записей (records).

        `refresh=True` bypasses the loader's cache and re-fetches from the
        CBR API (used by the admin /api/refresh and /api/force-refresh
        endpoints) - previously this parameter didn't exist at all, so
        calling it always silently served cached data and, when passed as
        a keyword argument by the admin refresh endpoint, raised
        TypeError instead.
        """
        df = await (self.loader.refresh_data(period_days) if refresh else self.loader.load_data(period_days))
        empty_result = {"dates": [], "records": [], "data": []}
        for col in SUPPORTED_CURRENCIES:
            empty_result[short_code(col)] = []
            empty_result[col] = []

        if df is None or df.empty:
            return empty_result

        df_copy = df.copy()
        if "date" in df_copy.columns:
            df_copy["date_str"] = df_copy["date"].dt.strftime("%Y-%m-%d")
        else:
            df_copy["date_str"] = []

        result = dict(empty_result)
        result["dates"] = df_copy["date_str"].tolist()

        for col in SUPPORTED_CURRENCIES:
            if col in df_copy.columns:
                # pd.notna() alone lets a raw Infinity through (it isn't
                # NaN) straight into a JSON response that can't serialize
                # it - use the same isfinite check as `records` below.
                values = [
                    round(float(x), 4) if pd.notna(x) and math.isfinite(x) else None
                    for x in df_copy[col]
                ]
            else:
                values = []
            result[short_code(col)] = values
            result[col] = values

        # merge_asof's backward join in the loader (key_rate, so far the
        # only external feature merged onto the rate history) leaves NaN
        # for any row older than that feature's own coverage - which
        # to_dict() below passes straight through as a bare float NaN.
        # Starlette's default JSONResponse serializes with allow_nan=False
        # (the JSON spec doesn't allow NaN/Infinity), so a single NaN deep
        # in a long-period window 500'd the *entire* response - not just
        # that one field - which is exactly why the 30-day chart worked
        # while the page's own default 180-day load silently never
        # rendered a chart at all. The per-currency `values` lists above
        # were already doing this pd.notna() swap; `records`/`data` just
        # hadn't been getting the same treatment.
        records = [
            {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in record.items()
            }
            for record in df_copy.to_dict(orient="records")
        ]
        result["records"] = records
        result["data"] = records
        return result

    async def get_stats(self) -> Dict[str, Any]:
        """Расчет текущих показателей курсов и изменений за день для всех поддерживаемых валют."""
        df = await self.loader.load_data(365)

        defaults = {"usd_rate": 81.41, "eur_rate": 94.06, "cny_rate": 11.30, "gbp_rate": 109.50}
        stats: Dict[str, Any] = {}

        if df is None or df.empty:
            for col, default in defaults.items():
                short = short_code(col)
                stats[f"{short}_current"] = default
                stats[f"{short}_change"] = 0.0
            stats["total_records"] = 0
            return stats

        for col, default in defaults.items():
            short = short_code(col)
            if col in df.columns:
                curr = float(df[col].iloc[-1])
                prev = float(df[col].iloc[-2]) if len(df) > 1 else curr
            else:
                curr, prev = default, default
            stats[f"{short}_current"] = round(curr, 2)
            stats[f"{short}_change"] = round(curr - prev, 2)

        stats["total_records"] = len(df)
        return stats

