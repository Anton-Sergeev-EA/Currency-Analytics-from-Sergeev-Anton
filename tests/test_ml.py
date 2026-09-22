"""
Unit tests for the ML feature engineering and ensemble model, independent
of any real CBR data or network access - synthetic frames only, so these
run identically in CI and on a laptop with no internet, and fast (no
hyperparameter search: tune=False everywhere here).

These pin two things that are easy to silently break and hard to notice
from the outside (a backtest can still "run" and print a number even if
its features are subtly wrong):

1. Feature engineering must not leak the current day's own value into its
   own lag/rolling features - the one bug class that would make every
   backtest R2/MAPE number in this project meaningless without anyone
   noticing (see model_evaluation.py's docstring on why this matters).
2. EnsembleModel's public interface (fit/predict/predict_with_uncertainty)
   behaves sanely on a simple, fully synthetic, near-linear relationship -
   a model that can't recover a nearly-linear y=2x1-0.5x2 signal on clean
   synthetic data has no chance on noisy real exchange-rate data.
"""
import numpy as np
import pandas as pd

from src.infrastructure.ml.features.engineer import FeatureEngineer
from src.infrastructure.ml.models.ensemble import EnsembleModel


def _synthetic_frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    usd = 90 + np.cumsum(rng.normal(0, 0.3, n))
    eur = 98 + np.cumsum(rng.normal(0, 0.35, n))
    return pd.DataFrame({"date": dates, "usd_rate": usd, "eur_rate": eur})


def test_feature_engineer_creates_lag_rolling_momentum_and_calendar_columns():
    df = FeatureEngineer().create_features(_synthetic_frame())
    for col in (
        "usd_rate_lag_1", "usd_rate_rolling_mean_3", "usd_rate_rolling_std_7",
        "usd_rate_pct_change_3", "day_of_week", "day", "month",
        "eur_rate_usd_rate_cross_lag1",
    ):
        assert col in df.columns, f"expected feature column {col!r} is missing"


def test_feature_engineer_lag_features_do_not_leak_the_current_value():
    raw = _synthetic_frame()
    df = FeatureEngineer().create_features(raw).sort_values("date").reset_index(drop=True)
    # A lag-1 feature at row i must equal the raw value at row i-1, never
    # row i's own value - otherwise the model would be "predicting" a
    # value using itself, and every downstream accuracy number is fake.
    assert np.allclose(
        df["usd_rate_lag_1"].iloc[1:].to_numpy(),
        df["usd_rate"].iloc[:-1].to_numpy(),
    )


def test_feature_engineer_key_rate_is_filled_without_days_since_change_feature():
    raw = _synthetic_frame()
    raw["key_rate"] = 15.0
    raw.loc[raw.index[-10:], "key_rate"] = 16.0
    df = FeatureEngineer().create_features(raw)
    assert df["key_rate"].isna().sum() == 0
    # Removed after a real walk-forward backtest on live CBR data showed
    # it hurt accuracy (CNY stopped beating the naive baseline) - see the
    # comment in engineer.py and the project's git history. Pinned here so
    # a future re-add is a deliberate, tested decision, not an accident.
    assert "days_since_key_rate_change" not in df.columns


def test_ensemble_model_recovers_a_near_linear_signal():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = 2 * X["f1"] - 0.5 * X["f2"] + rng.normal(scale=0.05, size=200)

    model = EnsembleModel()
    model.fit(X.iloc[:150], y.iloc[:150], tune=False)
    preds = model.predict(X.iloc[150:])

    assert len(preds) == 50
    assert np.corrcoef(preds, y.iloc[150:])[0, 1] > 0.8


def test_ensemble_model_weights_are_a_normalized_distribution():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"f1": rng.normal(size=80), "f2": rng.normal(size=80)})
    y = X["f1"] + rng.normal(scale=0.1, size=80)

    model = EnsembleModel()
    model.fit(X, y, tune=False)  # tune=False -> equal weights (documented fallback)

    assert abs(sum(model._weights.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in model._weights.values())


def test_ensemble_model_uncertainty_bounds_bracket_the_point_prediction():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"f1": rng.normal(size=100)})
    y = X["f1"] * 3 + rng.normal(scale=0.2, size=100)

    model = EnsembleModel()
    model.fit(X, y, tune=False)
    point, (lower, upper) = model.predict_with_uncertainty(X.iloc[:10], n_iterations=15)

    assert (lower <= point + 1e-9).all()
    assert (upper >= point - 1e-9).all()


def test_ensemble_model_predict_before_fit_raises_clearly():
    model = EnsembleModel()
    try:
        model.predict(pd.DataFrame({"f1": [0.0]}))
        assert False, "expected RuntimeError for predict() before fit()/load()"
    except RuntimeError:
        pass
