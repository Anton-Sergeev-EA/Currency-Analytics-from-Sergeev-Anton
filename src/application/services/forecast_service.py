import os
import pandas as pd
from typing import Dict, Any, Union, List
from src.core.constants import SUPPORTED_CURRENCIES, column_name, short_code
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.models.ensemble import EnsembleModel
from src.infrastructure.ml.features.engineer import FeatureEngineer
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class ForecastService:
    """Сервис прогнозирования курсов валют с ML-ансамблем и трендовым fallback-механизмом."""

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()

    async def get_forecast_with_uncertainty(self, days: int = 7) -> Dict[str, Any]:
        """Метод для интерфейсных таблиц и графиков."""
        return await self._generate_forecast(days=days, currency="all")

    async def get_forecast(self, days: int = 7, currency: str = "all") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Метод для RAG-ассистента и API."""
        return await self._generate_forecast(days=days, currency=currency)

    async def get_persistence_baseline(self, days: int = 7, currency: str = "usd_rate") -> List[Dict[str, Any]]:
        """
        "Наивный" прогноз (persistence / random-walk): курс не меняется.

        Это стандартный бенчмарк в прогнозировании валютных курсов —
        курсы FX statistically близки к случайному блужданию, поэтому
        "завтра = сегодня" на коротких горизонтах на удивление сложно
        обыграть. Используется как контрольная группа B в A/B-тесте
        (see src/ab_testing/ab_service.py), чтобы честно показать,
        действительно ли ансамблевая ML-модель (группа A) даёт
        измеримое улучшение — без необходимости держать в памяти
        вторую полноценную ML-модель.

        Доверительный интервал строится из реальной исторической
        волатильности (std дневных изменений), а не из произвольной
        константы.
        """
        df = await self.data_loader.load_data()
        if df is None or df.empty:
            return []

        col = column_name(currency)
        if col not in df.columns:
            return []

        last_val = float(df[col].iloc[-1])
        daily_returns = df[col].diff().dropna()
        daily_std = float(daily_returns.std()) if len(daily_returns) > 1 else last_val * 0.005
        last_date = pd.to_datetime(df["date"].iloc[-1]) if "date" in df.columns else pd.Timestamp.now()

        predictions = []
        for i in range(days):
            next_date = last_date + pd.Timedelta(days=i + 1)
            # Uncertainty widens with sqrt(horizon), as for a random walk.
            uncertainty = round(daily_std * ((i + 1) ** 0.5) * 1.96, 2)
            predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "day": i + 1,
                "rate": round(last_val, 2),
                "forecast": round(last_val, 2),
                "value": round(last_val, 2),
                "lower_bound": round(last_val - uncertainty, 2),
                "upper_bound": round(last_val + uncertainty, 2),
                "lower": round(last_val - uncertainty, 2),
                "upper": round(last_val + uncertainty, 2),
            })
        return predictions

    async def _generate_forecast(self, days: int = 7, currency: str = "all") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        df = await self.data_loader.load_data()
        if df is None or df.empty:
            logger.error("No historical data available for forecasting.")
            return {"error": "No historical data available"}

        c_lower = currency.lower().strip()
        if c_lower in ("all", ""):
            target_currencies = [c for c in SUPPORTED_CURRENCIES if c in df.columns]
        else:
            col = column_name(c_lower)
            target_currencies = [col] if col in df.columns else []

        if not target_currencies:
            return {"error": f"Unknown or unavailable currency: {currency}"}

        results = {}
        last_date = pd.to_datetime(df["date"].iloc[-1]) if "date" in df.columns else pd.Timestamp.now()
        non_target_cols = ["date"] + SUPPORTED_CURRENCIES

        for curr in target_currencies:
            # Проверяем возможные имена файлов моделей
            model_path = os.path.join(self.models_dir, f"{curr}_model.joblib")
            alt_path = os.path.join(self.models_dir, f"{short_code(curr)}_model.joblib")

            actual_path = model_path if os.path.exists(model_path) else (alt_path if os.path.exists(alt_path) else None)

            if not actual_path:
                logger.warning(f"ML model for {curr} not found. Using robust statistical trend fallback.")
                results[curr] = self._fallback_forecast(df, curr, days, last_date)
                continue

            model = EnsembleModel()
            try:
                model.load(actual_path)
            except Exception as e:
                logger.error(f"Failed to load ML model {actual_path}: {e}. Using fallback.")
                results[curr] = self._fallback_forecast(df, curr, days, last_date)
                continue

            curr_df = df.copy()
            predictions = []

            for i in range(days):
                try:
                    df_features = self.feature_engineer.create_features(curr_df)
                    feature_cols = [c for c in df_features.columns if c not in non_target_cols]
                    last_row = df_features[feature_cols].iloc[[-1]]
                    # predict() returns point estimates only - the
                    # confidence interval comes from
                    # predict_with_uncertainty(), a *separate* call
                    # (bootstrap over stored residuals). The previous
                    # version unpacked predict()'s single array as
                    # `mean_pred, lower, upper`, which would raise
                    # "too many values to unpack" the moment a real
                    # trained model existed on disk - it never
                    # surfaced only because no .joblib model has been
                    # trained yet, so this branch was never actually
                    # exercised.
                    mean_pred, (lower, upper) = model.predict_with_uncertainty(last_row)

                    pred_val = round(float(mean_pred[0]), 2)
                    low_val = round(float(lower[0]), 2)
                    up_val = round(float(upper[0]), 2)
                except Exception as e:
                    logger.error(f"Prediction step error for {curr}: {e}")
                    results[curr] = self._fallback_forecast(df, curr, days, last_date)
                    break

                next_date = last_date + pd.Timedelta(days=i + 1)
                date_str = next_date.strftime("%Y-%m-%d")

                # Универсальная структура элемента для любого фронтенда
                predictions.append({
                    "date": date_str,
                    "day": i + 1,
                    "rate": pred_val,
                    "forecast": pred_val,
                    "value": pred_val,
                    "lower_bound": low_val,
                    "upper_bound": up_val,
                    "lower": low_val,
                    "upper": up_val
                })

                new_row = {"date": next_date, curr: pred_val}
                for other_curr in SUPPORTED_CURRENCIES:
                    if other_curr != curr and other_curr in curr_df.columns:
                        new_row[other_curr] = curr_df[other_curr].iloc[-1]

                curr_df = pd.concat([curr_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                results[curr] = predictions

        # Если запрошена одна валюта, отдаем массив, если несколько — словарь
        # с коротким кодом, полным именем колонки и заглавными буквами на
        # каждую валюту (для обратной совместимости фронтенда/интеграций).
        if len(target_currencies) == 1:
            return results.get(target_currencies[0], [])

        output: Dict[str, Any] = {}
        for curr in target_currencies:
            preds = results.get(curr, [])
            short = short_code(curr)
            output[short] = preds
            output[curr] = preds
            output[short.upper()] = preds

        primary = output.get("usd") or next((output[short_code(c)] for c in target_currencies if output.get(short_code(c))), [])
        output["dates"] = [item["date"] for item in primary]
        return output

    def _fallback_forecast(self, df: pd.DataFrame, col: str, days: int, last_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Статистический расчет тренда при отсутствии или сбое файла .joblib."""
        if col not in df.columns:
            col = column_name(col)
        if col not in df.columns:
            return []

        last_val = float(df[col].iloc[-1])
        prev_val = float(df[col].iloc[-5]) if len(df) >= 5 else float(df[col].iloc[0])
        daily_trend = (last_val - prev_val) / max(min(len(df), 5), 1)

        predictions = []
        for i in range(days):
            next_date = last_date + pd.Timedelta(days=i + 1)
            date_str = next_date.strftime("%Y-%m-%d")

            pred_val = round(last_val + daily_trend * (i + 1), 2)
            uncertainty = round(0.45 * (i + 1), 2)

            predictions.append({
                "date": date_str,
                "day": i + 1,
                "rate": pred_val,
                "forecast": pred_val,
                "value": pred_val,
                "lower_bound": round(pred_val - uncertainty, 2),
                "upper_bound": round(pred_val + uncertainty, 2),
                "lower": round(pred_val - uncertainty, 2),
                "upper": round(pred_val + uncertainty, 2)
            })
        return predictions
