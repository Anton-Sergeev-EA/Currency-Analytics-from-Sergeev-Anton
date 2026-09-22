"""
Unit tests for src/infrastructure/ml/tracking.py.

The one thing that matters most here: this module must NEVER raise, no
matter what goes wrong on the MLflow side (disabled, unreachable,
mlflow itself broken) - model training must not fail because an optional
metrics dashboard couldn't be written to. Every test below monkeypatches
settings.MLFLOW_TRACKING_URI directly (not via .env) so it's independent
of whatever happens to be in the environment running the tests.
"""
from src.core.config import settings
from src.infrastructure.ml.tracking import log_backtest_run, log_training_run


class _FakeModel:
    """Just enough of EnsembleModel's public surface for log_training_run."""

    def __init__(self):
        self.n_train_rows = 733
        self.weights = {"lightgbm": 0.5, "naive_persistence": 0.5}
        self.cv_mae_ = {"lightgbm": 1.23, "naive_persistence": 0.98}
        self.tuned_params_ = {"lightgbm": {"n_estimators": 100}}


def test_log_training_run_is_a_noop_when_tracking_disabled(monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "")
    # Should return cleanly without even trying to import/call mlflow.
    log_training_run("usd_rate", _FakeModel())


def test_log_backtest_run_is_a_noop_when_tracking_disabled(monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "")
    log_backtest_run("usd_rate", {"available": True, "rmse": 1.0, "mae": 0.8, "mape": 1.0, "r2": 0.5, "baseline": {}})


def test_log_backtest_run_is_a_noop_when_metrics_unavailable(monkeypatch):
    # Tracking URI set, but the backtest itself didn't produce metrics -
    # nothing to log, and this must not KeyError on missing rmse/mae/etc.
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    log_backtest_run("usd_rate", {"available": False, "reason": "not enough data"})


def test_log_training_run_never_raises_when_server_unreachable(monkeypatch):
    # Port 1 on localhost: nothing listens there, so this exercises the
    # real "mlflow is configured but unreachable" path end-to-end,
    # through the actual mlflow client, not a mock.
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    log_training_run("usd_rate", _FakeModel())  # must not raise


def test_log_backtest_run_never_raises_when_server_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    log_backtest_run(
        "usd_rate",
        {
            "available": True,
            "rmse": 1.0,
            "mae": 0.8,
            "mape": 1.0,
            "r2": 0.5,
            "test_days": 20,
            "windows_evaluated": 3,
            "beats_naive_baseline": False,
            "baseline": {"rmse": 0.9, "r2": 0.6},
        },
    )  # must not raise
