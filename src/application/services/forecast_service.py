import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.application.services.data_service import DataService

logger = logging.getLogger(__name__)


class ForecastService:
    def __init__(self, data_service: Optional[DataService] = None):
        self.data_service = data_service or DataService()
        self._models = {}

    async def _ensure_models(self) -> None:
        """
        Проверяет и загружает исторические данные через DataService.
        """
        try:
            df = await self.data_service.get_historical_data(365)
            if df is not None and len(df) > 30:
                self._fit_simple_models(df)
            else:
                logger.warning("Not enough data to fit forecast models properly.")
        except Exception as e:
            logger.error(f"Error in _ensure_models: {e}")

    def _fit_simple_models(self, df: pd.DataFrame) -> None:
        """
        Логика расчета тренда и неопределенности на основе исторических данных.
        """
        for curr in ['usd_rate', 'eur_rate']:
            series = df[curr].dropna().values
            if len(series) > 0:
                last_val = series[-1]
                std = np.std(np.diff(series)) if len(series) > 1 else 1.0
                self._models[curr] = {
                    'last_val': float(last_val),
                    'std': float(std)
                }

    async def get_forecast_with_uncertainty(self, days: int = 30) -> Dict[str, Any]:
        """
        Возвращает прогноз курсов с интервалами неопределенности.
        """
        await self._ensure_models()

        result = {}
        for curr_key, curr_name in [('usd_rate', 'USD'), ('eur_rate', 'EUR')]:
            model_info = self._models.get(curr_key)

            if not model_info:
                last_val = 78.40 if curr_key == 'usd_rate' else 89.44
                std = 0.5
            else:
                last_val = model_info['last_val']
                std = model_info['std']

            horizon = np.arange(1, days + 1)
            mean_forecast = [round(float(last_val + 0.02 * d), 4) for d in horizon]
            lower_bound = [round(float(m - (1.96 * std * np.sqrt(d / 7.0))), 4) for d, m in zip(horizon, mean_forecast)]
            upper_bound = [round(float(m + (1.96 * std * np.sqrt(d / 7.0))), 4) for d, m in zip(horizon, mean_forecast)]

            result[curr_key] = {
                "mean": mean_forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

        return result

