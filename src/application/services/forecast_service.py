import os
import pandas as pd
from typing import Dict, Any, Union, List
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

    async def _generate_forecast(self, days: int = 7, currency: str = "all") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        df = await self.data_loader.load_data()
        if df is None or df.empty:
            logger.error("No historical data available for forecasting.")
            return {"error": "No historical data available"}

        target_currencies = []
        c_lower = currency.lower().strip()
        if c_lower in ["usd", "usd_rate"]:
            target_currencies = ["usd_rate"]
        elif c_lower in ["eur", "eur_rate"]:
            target_currencies = ["eur_rate"]
        else:
            target_currencies = ["usd_rate", "eur_rate"]

        results = {}
        last_date = pd.to_datetime(df["date"].iloc[-1]) if "date" in df.columns else pd.Timestamp.now()

        for curr in target_currencies:
            # Проверяем возможные имена файлов моделей
            model_path = os.path.join(self.models_dir, f"{curr}_model.joblib")
            alt_path = os.path.join(self.models_dir, f"{curr.replace('_rate', '')}_model.joblib")

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
                    feature_cols = [c for c in df_features.columns if c not in ["date", "usd_rate", "eur_rate"]]
                    last_row = df_features[feature_cols].iloc[[-1]]
                    mean_pred, lower, upper = model.predict(last_row)

                    pred_val = round(float(mean_pred[0]), 2)
                    low_val = round(float(lower[0]), 2)
                    up_val = round(float(upper[0]), 2)
                except Exception as e:
                    logger.error(f"Prediction step error for {curr}: {e}")
                    return self._fallback_forecast(df, curr, days, last_date)

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
                other_curr = "eur_rate" if curr == "usd_rate" else "usd_rate"
                if other_curr in curr_df.columns:
                    new_row[other_curr] = curr_df[other_curr].iloc[-1]

                curr_df = pd.concat([curr_df, pd.DataFrame([new_row])], ignore_index=True)

            results[curr] = predictions

        # Если запрошена одна валюта, отдаем массив, если все — словарь со всеми алиасами ключей
        if len(target_currencies) == 1:
            return results[target_currencies[0]]

        usd_list = results.get("usd_rate", [])
        eur_list = results.get("eur_rate", [])

        return {
            "usd": usd_list,
            "usd_rate": usd_list,
            "USD": usd_list,
            "eur": eur_list,
            "eur_rate": eur_list,
            "EUR": eur_list,
            "dates": [item["date"] for item in usd_list] if usd_list else []
        }

    def _fallback_forecast(self, df: pd.DataFrame, col: str, days: int, last_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Статистический расчет тренда при отсутствии или сбое файла .joblib."""
        if col not in df.columns:
            col = "usd_rate" if "usd" in col else "eur_rate"

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
