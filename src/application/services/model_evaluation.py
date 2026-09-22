"""
ModelEvaluator — real, held-out backtest metrics for the monitoring dashboard.

`/monitoring/api/model-accuracy` and `/monitoring/api/prediction-test` used
to return `random.random()`-perturbed numbers "to look realistic". This
replaces that with an actual walk-forward backtest: train a model on all
data *except* the last `test_days`, predict those held-out days, and
compare to what actually happened. That is genuinely out-of-sample (the
production model in data/models/ is trained on the full dataset, so
evaluating it on its own training data would be in-sample and
overstate accuracy).

This is real CPU work (fitting 4 small tree-based regressors on a few
hundred rows), which is cheap here but not free, so results are cached
with a TTL - the dashboard doesn't need this recomputed on every poll,
and a 4GB-RAM VDS shouldn't be asked to.
"""
import time
from typing import Any, Dict

import numpy as np

from src.core.constants import DEFAULT_TRAINING_WINDOW_DAYS, SUPPORTED_CURRENCIES, short_code
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.features.engineer import FeatureEngineer
from src.infrastructure.ml.models.ensemble import EnsembleModel
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 6 * 3600  # recompute at most every 6 hours.
TEST_DAYS = 20
MIN_TRAIN_ROWS = 30


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """RMSE/MAE/MAPE/R2 for one set of (actual, predicted) pairs - used for
    both the model's own backtest and the naive baseline it's compared
    against, so the two numbers are computed exactly the same way."""
    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / actual)) * 100)
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 2), "r2": round(r2, 4)}


class ModelEvaluator:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self._cache: Dict[str, tuple] = {}  # currency -> (computed_at, metrics)

    async def get_accuracy(self, currency: str = "usd_rate") -> Dict[str, Any]:
        cached = self._cache.get(currency)
        if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
            return cached[1]

        metrics = await self._compute_backtest(currency)
        self._cache[currency] = (time.time(), metrics)
        return metrics

    async def _compute_backtest(self, currency: str) -> Dict[str, Any]:
        df = await self.data_loader.load_data(DEFAULT_TRAINING_WINDOW_DAYS)
        if df is None or df.empty or currency not in df.columns:
            return {"available": False, "reason": "no historical data"}

        df_features = self.feature_engineer.create_features(df)
        feature_cols = [c for c in df_features.columns if c not in ["date"] + SUPPORTED_CURRENCIES]
        clean = df_features.dropna(subset=feature_cols + [currency]).reset_index(drop=True)

        if len(clean) < MIN_TRAIN_ROWS + TEST_DAYS:
            return {"available": False, "reason": "not enough history for a held-out backtest yet"}

        train = clean.iloc[:-TEST_DAYS]
        test = clean.iloc[-TEST_DAYS:]

        try:
            model = EnsembleModel()
            model.fit(train[feature_cols], train[currency])
            preds = model.predict(test[feature_cols])
        except Exception as exc:
            logger.error("Backtest training failed for %s: %s", currency, exc, exc_info=True)
            return {"available": False, "reason": f"backtest training failed: {exc}"}

        actual = test[currency].to_numpy()

        # The model's own held-out error.
        model_metrics = _regression_metrics(actual, preds)

        # Naive persistence baseline: "tomorrow's rate = today's rate".
        # For a series this close to a random walk, this is a genuinely
        # hard baseline to beat, and a small MAPE alone doesn't tell you
        # whether the model beat it - only comparing against it does.
        # clean[currency] (not just train/test) so the very first held-out
        # day still has a real previous value to persist from.
        naive_preds = clean[currency].shift(1).iloc[-TEST_DAYS:].to_numpy()
        baseline_metrics = _regression_metrics(actual, naive_preds)

        return {
            "available": True,
            "method": "walk-forward: trained on all data except the last "
                      f"{TEST_DAYS} days, evaluated only on those held-out days",
            "test_days": TEST_DAYS,
            **model_metrics,
            "baseline": {
                "description": "naive persistence (today's rate used as tomorrow's prediction)",
                **baseline_metrics,
            },
            "beats_naive_baseline": model_metrics["rmse"] < baseline_metrics["rmse"],
        }

    async def get_current_predictions(self) -> Dict[str, Any]:
        """One-step-ahead prediction from the *production* model (or its
        trend fallback), for a live sanity check on the dashboard - not a
        random number."""
        from src.application.services.forecast_service import ForecastService

        service = ForecastService()
        result = {}
        for ccy in SUPPORTED_CURRENCIES:
            try:
                forecast = await service.get_forecast(days=1, currency=ccy)
                if forecast:
                    result[short_code(ccy)] = forecast[0]["rate"]
            except Exception as exc:
                logger.warning("Prediction test failed for %s: %s", ccy, exc)
        return result
