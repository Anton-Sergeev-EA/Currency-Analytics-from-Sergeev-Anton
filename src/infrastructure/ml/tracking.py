"""
MLflow experiment tracking for model training and backtesting.

The MLflow tracking server itself lives outside this repository - it runs
as its own systemd service on the VDS, with its own venv and its own
sqlite backing store (see README's "Monitoring" section for the SSH
tunnel instructions to view it). This module only *writes* to it.

Design choices:

- Every public function here is a no-op, not a crash, if MLflow logging
  can't happen - whether that's because settings.MLFLOW_TRACKING_URI is
  blank (logging deliberately disabled, e.g. local dev with no server
  running) or because the configured server is unreachable (network
  hiccup, server down for maintenance). Model training must never fail,
  or silently produce no models, just because an optional metrics
  dashboard couldn't be reached - see train_models.py, which calls these
  functions after each currency's training/backtest already succeeded.

- Run names match what a human browsing the MLflow UI would expect:
  "train-{currency}" (e.g. "train-usd_rate") for a training run and
  "backtest-{short_code}" (e.g. "backtest-usd") for a backtest run - one
  pair per currency, per invocation of train_models.py.

- The `mlflow` import itself is lazy (inside the functions, not at module
  level) so that importing this module - and therefore train_models.py -
  never fails even if the mlflow-skinny package somehow isn't installed
  in a given environment; the calling code degrades to "logging skipped"
  the same way it does for a network failure.
"""
import os
from typing import Any, Dict

from src.common.logger.logger import get_logger
from src.core.config import settings

logger = get_logger(__name__)

# mlflow's HTTP client retries a failed request several times with
# backoff by default - against an unreachable host that adds up to
# minutes, not seconds, which would turn "MLflow is down" into "model
# training hangs". setdefault() so an operator can still override these
# via the real environment if they want different behavior; this only
# fills them in when nothing else already has.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
# This container's image deliberately doesn't ship a git binary (it's an
# inference/training runtime, not a dev environment) - mlflow's autolog
# tries to capture git SHA/branch/remote for each run regardless, and
# without git that's four verbose warning blocks per run, not an error.
# Silencing rather than installing git: cheaper than an extra apt-get
# layer just to make an optional metadata field available.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def _tracking_enabled() -> bool:
    return bool(settings.MLFLOW_TRACKING_URI.strip())


def log_training_run(currency: str, model: Any) -> None:
    """
    Log one "train-{currency}" run: how the ensemble was fit.

    `model` is a fitted EnsembleModel. Reads only its public surface
    (weights, cv_mae_, tuned_params_, n_train_rows) - see ensemble.py and
    trainer.py for where those are populated.
    """
    if not _tracking_enabled():
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"train-{currency}"):
            mlflow.set_tag("target", currency)
            mlflow.log_param("target", currency)
            mlflow.log_param("train_rows", getattr(model, "n_train_rows", None))
            mlflow.log_param("candidate_models", ",".join(sorted(model.weights.keys())))

            for name, weight in model.weights.items():
                mlflow.log_metric(f"weight_{name}", weight)
            for name, cv_mae in model.cv_mae_.items():
                mlflow.log_metric(f"cv_mae_{name}", cv_mae)

            for name, params in model.tuned_params_.items():
                for param_name, value in params.items():
                    # MLflow params are strings; hyperparameter values
                    # are ints/floats/None, so just str() them rather
                    # than trying to preserve type through the API.
                    mlflow.log_param(f"{name}__{param_name}", str(value))

    except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
        logger.warning("MLflow logging skipped for train-%s: %s", currency, e)


def log_backtest_run(currency: str, metrics: Dict[str, Any]) -> None:
    """
    Log one "backtest-{short_code}" run: the walk-forward backtest result
    from ModelEvaluator.get_accuracy() (see model_evaluation.py) - the
    same averaged-over-N-windows metrics train_models.py already prints
    to the console and the monitoring dashboard already shows.
    """
    if not _tracking_enabled():
        return
    if not metrics.get("available"):
        return

    short_code = currency.replace("_rate", "")

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"backtest-{short_code}"):
            mlflow.set_tag("target", currency)
            mlflow.log_param("target", currency)
            mlflow.log_param("test_days", metrics.get("test_days"))
            mlflow.log_param("windows_evaluated", metrics.get("windows_evaluated"))
            mlflow.log_param("beats_naive_baseline", metrics.get("beats_naive_baseline"))

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])
            mlflow.log_metric("r2", metrics["r2"])

            baseline = metrics.get("baseline") or {}
            if "rmse" in baseline:
                mlflow.log_metric("baseline_rmse", baseline["rmse"])
            if "r2" in baseline:
                mlflow.log_metric("baseline_r2", baseline["r2"])

    except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
        logger.warning("MLflow logging skipped for backtest-%s: %s", short_code, e)
