"""
Regression tests for the specific incident that motivated this file: the
monitoring dashboard hanging on "Loading..." after a deploy, because
`/monitoring/api/model-accuracy` used to run several minutes of CPU-bound
model fitting inline inside an `async def` route, on an app that runs a
single Uvicorn worker (see the Dockerfile) - blocking that worker's event
loop, and therefore every other endpoint, for as long as it took.

The fix has two independent, testable parts:

1. The CPU-bound half of the backtest (`ModelEvaluator._run_backtest_windows`)
   is a plain synchronous function with no `await` inside it - safe to
   hand to `asyncio.to_thread` without ever blocking the event loop, no
   matter how long it takes.
2. `ModelEvaluator.get_accuracy()` de-duplicates concurrent callers for
   the same currency via a per-currency lock, so a cold cache plus two
   near-simultaneous requests (e.g. a real visitor racing the startup
   warm-up task) doesn't pay the expensive computation twice in parallel.

Both are exercised here on small synthetic data, fast enough to run in
CI - not a real walk-forward backtest on real currencies (that needs
network access to the Bank of Russia and belongs on the VDS, per
train_models.py).
"""
import asyncio

import numpy as np
import pandas as pd

from src.application.services.model_evaluation import MIN_TRAIN_ROWS, TEST_DAYS, ModelEvaluator


def _synthetic_clean_frame(n: int = 100) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    target = "usd_rate"
    level = 80 + np.cumsum(rng.normal(0, 0.2, n))
    df = pd.DataFrame({
        target: level,
        f"{target}_lag_1": np.concatenate([[level[0]], level[:-1]]),
        "other_feature": rng.normal(size=n),
    })
    feature_cols = [f"{target}_lag_1", "other_feature"]
    return df, feature_cols


def test_run_backtest_windows_is_a_plain_synchronous_function():
    """Pins the contract asyncio.to_thread relies on: no coroutine, no
    event-loop access, callable directly from a worker thread."""
    clean, feature_cols = _synthetic_clean_frame(n=100)
    assert not asyncio.iscoroutinefunction(ModelEvaluator._run_backtest_windows)

    model_metrics, baseline_metrics = ModelEvaluator._run_backtest_windows(
        clean, feature_cols, "usd_rate", n_windows=1
    )

    assert len(model_metrics) == 1
    assert len(baseline_metrics) == 1
    for metrics in (model_metrics[0], baseline_metrics[0]):
        for key in ("rmse", "mae", "mape", "r2"):
            assert key in metrics


def test_run_backtest_windows_respects_min_train_rows_and_test_days():
    # Just enough rows for exactly one window - a regression guard on the
    # window-sizing arithmetic in _compute_backtest/_run_backtest_windows.
    clean, feature_cols = _synthetic_clean_frame(n=MIN_TRAIN_ROWS + TEST_DAYS)
    model_metrics, _ = ModelEvaluator._run_backtest_windows(
        clean, feature_cols, "usd_rate", n_windows=3
    )
    # Only one window fits without dropping below MIN_TRAIN_ROWS.
    assert len(model_metrics) == 1


def test_get_accuracy_deduplicates_concurrent_callers_for_the_same_currency():
    evaluator = ModelEvaluator()
    call_count = 0

    async def fake_compute_backtest(currency):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)  # long enough for both callers to race in
        return {"available": True, "rmse": 1.0, "mae": 1.0, "mape": 1.0, "r2": 0.5,
                "baseline": {"rmse": 1.0}, "beats_naive_baseline": False}

    evaluator._compute_backtest = fake_compute_backtest

    async def run_both():
        return await asyncio.gather(
            evaluator.get_accuracy("usd_rate"),
            evaluator.get_accuracy("usd_rate"),
        )

    results = asyncio.run(run_both())

    assert call_count == 1, "both concurrent callers should share one computation, not run it twice"
    assert results[0] == results[1]
