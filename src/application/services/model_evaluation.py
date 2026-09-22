"""
ModelEvaluator — real, held-out backtest metrics for the monitoring dashboard.

`/monitoring/api/model-accuracy` and `/monitoring/api/prediction-test` used
to return `random.random()`-perturbed numbers "to look realistic". This
replaces that with an actual walk-forward backtest: train a model on all
data *before* a held-out window, predict that window, and compare to what
actually happened. That is genuinely out-of-sample (the production model
in data/models/ is trained on the full dataset, so evaluating it on its
own training data would be in-sample and overstate accuracy).

The backtest averages over several non-overlapping held-out windows
(walking backward from the most recent data), not just the last one. A
single 20-day slice is noisy enough on its own to flip a verdict purely
by luck: this project's own history has a real example of it (the CNY
model briefly "beat" the naive baseline, then "lost" to it, between two
runs on nearly the same real data and only a small code change - the
kind of instability a single held-out window can't tell apart from a
genuine regression). Averaging several windows doesn't remove that noise,
but it does make one unlucky window much less likely to flip the
headline verdict on its own.

This is real CPU work (fitting 4 small tree-based regressors, with a
bounded hyperparameter search on the largest window), which is cheap here
but not free, so results are cached with a TTL - the dashboard doesn't
need this recomputed on every poll, and a 4GB-RAM VDS shouldn't be asked
to. To keep the extra windows cheap on modest hardware, only the most
recent (largest-train-set) window runs the hyperparameter search; earlier
windows fit each base model with plain defaults, which is enough to
sanity-check the model's stability without tripling the tuning cost.

That CPU work runs in a worker thread (`asyncio.to_thread`), never inline
in an `async def`. This project runs a single Uvicorn worker by default
(see the Dockerfile - deliberate, to fit a 4GB VDS), so CPU-bound code
awaited directly inside a route handler blocks that one worker's entire
event loop: every other endpoint - health checks, live forecasts,
everything - goes unresponsive for as long as the computation takes,
which for 4 currencies with a cold cache is minutes, not seconds. This
is not a hypothetical: it is exactly what made `/monitoring/dashboard`
itself appear to hang on a fresh deploy. A per-currency lock in
`get_accuracy()` also makes sure a cold cache plus two callers at once
(a real visitor racing the startup warm-up below, or two visitors at
once) doesn't pay that cost twice in parallel on a memory-capped VDS.
`src/main.py`'s startup hook fires this computation for every currency
in the background right after boot, so in normal operation a real
visitor almost never hits the cold path at all - they land on an
already-warm 6-hour cache.
"""
import asyncio
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.constants import DEFAULT_TRAINING_WINDOW_DAYS, SUPPORTED_CURRENCIES, short_code
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.features.engineer import FeatureEngineer
from src.infrastructure.ml.models.ensemble import EnsembleModel
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 6 * 3600  # recompute at most every 6 hours.
TEST_DAYS = 20
N_WINDOWS = 3  # non-overlapping held-out windows, walking backward from the most recent data.
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


def _average_metrics(window_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across windows - a simple, transparent way to
    combine them. Not a pooled/weighted average, so each window (even a
    shorter early one, if any) counts equally."""
    keys = ("rmse", "mae", "mape", "r2")
    decimals = {"mape": 2}
    return {
        key: round(float(np.mean([m[key] for m in window_metrics])), decimals.get(key, 4))
        for key in keys
    }


class ModelEvaluator:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self._cache: Dict[str, tuple] = {}  # currency -> (computed_at, metrics)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, currency: str) -> asyncio.Lock:
        lock = self._locks.get(currency)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[currency] = lock
        return lock

    async def get_accuracy(self, currency: str = "usd_rate") -> Dict[str, Any]:
        cached = self._cache.get(currency)
        if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
            return cached[1]

        # One computation per currency at a time - without this, a cold
        # cache plus two near-simultaneous callers (e.g. a real visitor
        # racing the startup warm-up task) would both pay the multi-minute
        # tuning cost at once on a memory-capped VDS.
        async with self._lock_for(currency):
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

        # How many non-overlapping TEST_DAYS windows fit while still
        # leaving MIN_TRAIN_ROWS for the earliest (smallest-train-set) one.
        max_windows = (len(clean) - MIN_TRAIN_ROWS) // TEST_DAYS
        n_windows = max(1, min(N_WINDOWS, max_windows))

        try:
            # CPU-bound (model fitting + hyperparameter search) with no
            # I/O - run off the event loop in a worker thread. See this
            # module's docstring for why that matters on a single-worker
            # deployment.
            window_model_metrics, window_baseline_metrics = await asyncio.to_thread(
                self._run_backtest_windows, clean, feature_cols, currency, n_windows
            )
        except Exception as exc:
            logger.error("Backtest training failed for %s: %s", currency, exc, exc_info=True)
            return {"available": False, "reason": f"backtest training failed: {exc}"}

        if not window_model_metrics:
            return {"available": False, "reason": "not enough history for a held-out backtest yet"}

        model_metrics = _average_metrics(window_model_metrics)
        baseline_metrics = _average_metrics(window_baseline_metrics)
        windows_evaluated = len(window_model_metrics)

        return {
            "available": True,
            "method": f"walk-forward: averaged over {windows_evaluated} non-overlapping "
                      f"{TEST_DAYS}-day held-out window(s), each trained only on data before it",
            "test_days": TEST_DAYS,
            "windows_evaluated": windows_evaluated,
            **model_metrics,
            "baseline": {
                "description": "naive persistence (today's rate used as tomorrow's prediction)",
                **baseline_metrics,
            },
            "beats_naive_baseline": model_metrics["rmse"] < baseline_metrics["rmse"],
        }

    @staticmethod
    def _run_backtest_windows(
        clean, feature_cols: List[str], currency: str, n_windows: int
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """The synchronous, CPU-bound half of `_compute_backtest` - no I/O,
        no `await`, so it's safe to run in a worker thread via
        `asyncio.to_thread` without touching the event loop."""
        window_model_metrics: List[Dict[str, float]] = []
        window_baseline_metrics: List[Dict[str, float]] = []

        for w in range(n_windows):
            end = len(clean) - w * TEST_DAYS
            start = end - TEST_DAYS
            train = clean.iloc[:start]
            test = clean.iloc[start:end]
            if len(train) < MIN_TRAIN_ROWS:
                break

            # Only the most recent (largest-train-set) window pays for
            # the hyperparameter search; earlier windows use plain
            # defaults so N_WINDOWS backtests don't cost N_WINDOWS times
            # the tuning budget on a memory-capped VDS.
            #
            # Same naive-persistence ensemble candidate as production
            # training (see ensemble.py's module docstring) - the
            # backtest should honestly reflect what actually gets
            # deployed, not a version of the model with a candidate
            # quietly left out.
            lag1_col = f"{currency}_lag_1"
            model = EnsembleModel(
                target_lag1_column=lag1_col if lag1_col in feature_cols else None
            )
            model.fit(train[feature_cols], train[currency], tune=(w == 0))
            preds = model.predict(test[feature_cols])
            actual = test[currency].to_numpy()

            window_model_metrics.append(_regression_metrics(actual, preds))

            naive_preds = clean[currency].shift(1).iloc[start:end].to_numpy()
            window_baseline_metrics.append(_regression_metrics(actual, naive_preds))

        return window_model_metrics, window_baseline_metrics

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
