"""
EnsembleModel — combines LightGBM, XGBoost, Random Forest, Gradient
Boosting, and (when available) a naive-persistence "model" into a single
weighted predictor with bootstrap-based uncertainty estimates.

This class was referenced (imported and called) by `ModelTrainer`,
`ForecastService`, and the A/B testing service, but did not exist
anywhere in the codebase or its git history — reconstructed here from the
exact interface those call sites already depend on:

    model = EnsembleModel(target_lag1_column="usd_rate_lag_1")  # optional
    model.fit(X, y)
    model.predict(X) -> array of length len(X)
    model.predict_with_uncertainty(X, n_iterations=20) -> (mean, (lower, upper))
    model.save(path)
    EnsembleModel().load(path)

## Design notes

**Point prediction** is a weighted average of the base regressors'
predictions, not a plain mean. Each model's weight comes from its own
cross-validation error during tuning (see below) — a model that reliably
scored a lower CV MAE for this specific currency gets more say than one
that didn't, rather than every model always getting an equal vote
regardless of how well it actually did. Weights are `1 / CV_MAE`,
normalized to sum to 1. When tuning didn't run (see below), weights fall
back to equal (1 / number of models) — the old plain-average behavior.

**The naive-persistence candidate.** This project's own walk-forward
backtests (see `model_evaluation.py` and the README's "Model accuracy"
section) repeatedly showed that for day-ahead FX, a naive "tomorrow =
today" forecast is a genuinely hard benchmark that the 4 tree-based
models do not reliably beat — not a bug in this codebase, but the honest
reality of forecasting a series this close to a random walk from public
data alone. Rather than keep chasing that gap with more engineered
features (a real earlier attempt at that made things worse - see
engineer.py), the ensemble is given the naive forecast itself as a fifth,
zero-parameter candidate (`_NaivePersistenceModel`), weighted by the same
CV-performance rule as the other four. This is a standard technique in
forecasting (blending a naive/statistical baseline with ML models rather
than treating them as competitors - the winning entries of forecasting
competitions like the M4 routinely do this), not a way to disguise a
weak model as a strong one: when the naive forecast is genuinely the best
predictor available for a currency, the ensemble is now free to lean on
it instead of overweighting ML models that cannot reliably beat it: the
weighting is still earned by held-out CV performance, exactly like every
other candidate, and can end up anywhere from near-zero to dominant. It
is included only when the caller supplies `target_lag1_column` (the name
of the `{currency}_lag_1` feature already produced by
`FeatureEngineer`) - `ForecastService` doesn't need to pass it, since it
only ever calls `load()`, which restores whatever candidates (including
this one) were present when the model was trained and saved.

**Uncertainty estimation** uses residual bootstrap rather than refitting
models at prediction time (refitting several models x N iterations on
every forecast request would be far too slow for an API endpoint). During
`fit()`, in-sample residuals (true - predicted, using the same weighted
average as `predict()`) are computed once and stored. At prediction time,
`n_iterations` bootstrap samples are drawn from those residuals and added
to the point prediction; the 2.5th and 97.5th percentiles of the
resulting distribution form the (lower, upper) bound — a widely-used,
cheap approximation of a ~95% prediction interval. It is a simplification
(in-sample residuals tend to slightly underestimate true out-of-sample
uncertainty), documented here rather than hidden, so it can be swapped
for out-of-fold residuals later without changing any caller.

**Hyperparameter tuning**: each tree-based base model used to fit with
plain library defaults, which for a few hundred rows and 30+ correlated
lag/rolling features is a recipe for overfitting the training set and
doing no better than noise out of sample (the walk-forward backtest in
model_evaluation.py first surfaced this as a negative R2). `fit()` runs a
small, bounded `RandomizedSearchCV` per tree-based model with
`TimeSeriesSplit` folds instead of a random/shuffled split — shuffling
would let a fold "predict the past from the future", which is not a real
forecasting scenario. The search space and iteration count are kept
deliberately small (a handful of candidates x a few folds) so this stays
fast and memory-light enough for a small VDS; it is not an exhaustive
grid search. Each search's own held-out CV score is then reused as this
model's ensemble weight, so the tuning step pays for itself twice over.
The naive-persistence candidate has no hyperparameters to search, so it
gets a plain `TimeSeriesSplit` CV-MAE computation instead of
`RandomizedSearchCV`, over the same folds, for a like-for-like weight.
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Small, bounded search spaces - enough to move off library defaults
# (which tend to overfit a few hundred noisy rows) without turning every
# training run into a multi-minute grid search on a memory-capped VDS.
_PARAM_DISTRIBUTIONS = {
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [5, 10, 20],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 3, 5, 10],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_leaf": [1, 5, 10],
    },
}

# Below this many rows, TimeSeriesSplit folds get too small to mean
# anything - fall back to plain fit() with library defaults instead of
# tuning on noise.
_MIN_ROWS_TO_TUNE = 60
_TUNE_CV_SPLITS = 3
_TUNE_N_ITER = 8

NAIVE_MODEL_NAME = "naive_persistence"


class _NaivePersistenceModel:
    """A zero-parameter 'model': always predicts the most recent known
    value of the target itself (the `{currency}_lag_1` feature). Cannot
    overfit and needs no tuning - it exists so the ensemble can weight it
    against the real models on equal footing, per `EnsembleModel`'s
    module docstring above.

    Implements the minimal (get_params/set_params/fit/predict) surface so
    it can sit in the same `self._models` dict and go through joblib
    save/load like the sklearn/xgboost/lightgbm estimators around it,
    without needing to special-case it in most of EnsembleModel's code.
    """

    def __init__(self, lag1_column: str) -> None:
        self.lag1_column = lag1_column

    def get_params(self, deep: bool = True) -> dict:
        return {"lag1_column": self.lag1_column}

    def set_params(self, **params) -> "_NaivePersistenceModel":
        if "lag1_column" in params:
            self.lag1_column = params["lag1_column"]
        return self

    def fit(self, X: pd.DataFrame, y=None) -> "_NaivePersistenceModel":
        # Nothing to learn - "tomorrow = today" has no free parameters.
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag1_column].to_numpy()


class EnsembleModel:
    def __init__(self, random_state: int = 42, target_lag1_column: Optional[str] = None) -> None:
        self.random_state = random_state
        self.target_lag1_column = target_lag1_column
        self._models: dict[str, object] = {
            "lightgbm": LGBMRegressor(random_state=random_state, verbosity=-1),
            "xgboost": XGBRegressor(random_state=random_state, verbosity=0),
            "random_forest": RandomForestRegressor(random_state=random_state, n_jobs=1),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }
        if target_lag1_column is not None:
            self._models[NAIVE_MODEL_NAME] = _NaivePersistenceModel(target_lag1_column)
        self._feature_columns: Optional[list[str]] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False
        self.tuned_params_: dict[str, dict] = {}
        # Equal weights until fit() (re)computes them from CV performance.
        self._weights: dict[str, float] = {name: 1.0 / len(self._models) for name in self._models}

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = True) -> "EnsembleModel":
        logger.info("Fitting ensemble (%d candidate model(s)) on %d rows", len(self._models), len(X))
        self._feature_columns = list(X.columns)

        if self.target_lag1_column is not None and self.target_lag1_column not in X.columns:
            # Defensive: a caller passed a column name that isn't
            # actually in this X - drop the naive candidate rather than
            # KeyError deep inside predict() later on a production request.
            logger.warning(
                "target_lag1_column=%r not found in X - dropping the naive-persistence "
                "ensemble candidate for this fit().", self.target_lag1_column,
            )
            self._models.pop(NAIVE_MODEL_NAME, None)

        can_tune = tune and len(X) >= _MIN_ROWS_TO_TUNE
        n_splits = min(_TUNE_CV_SPLITS, max(2, len(X) // 30)) if can_tune else 0

        cv_mae_by_model: dict[str, float] = {}
        for name, model in self._models.items():
            if can_tune and name in _PARAM_DISTRIBUTIONS:
                search = RandomizedSearchCV(
                    model,
                    _PARAM_DISTRIBUTIONS[name],
                    n_iter=_TUNE_N_ITER,
                    cv=TimeSeriesSplit(n_splits=n_splits),
                    scoring="neg_mean_absolute_error",
                    random_state=self.random_state,
                    n_jobs=1,
                )
                search.fit(X, y)
                self._models[name] = search.best_estimator_
                self.tuned_params_[name] = search.best_params_
                cv_mae = -search.best_score_
                cv_mae_by_model[name] = cv_mae
                logger.info("Tuned %s: %s (CV MAE=%.4f)", name, search.best_params_, cv_mae)
            elif name == NAIVE_MODEL_NAME:
                # No hyperparameters to search - a plain walk-forward CV
                # MAE over the same fold structure, so its weight is
                # earned the same way as the tuned models' weights are.
                if can_tune:
                    fold_maes = []
                    for _, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
                        val_pred = model.predict(X.iloc[val_idx])
                        val_actual = y.iloc[val_idx].to_numpy()
                        fold_maes.append(float(np.mean(np.abs(val_actual - val_pred))))
                    cv_mae = float(np.mean(fold_maes))
                    cv_mae_by_model[name] = cv_mae
                    logger.info("Naive persistence CV MAE=%.4f (not tuned - no free parameters)", cv_mae)
                model.fit(X, y)
            else:
                model.fit(X, y)

        if cv_mae_by_model:
            # Inverse-error weighting: 1/MAE so a lower (better) error
            # yields a higher weight, then normalized to sum to 1. A tiny
            # epsilon guards against a division by an implausible exact-0
            # MAE (perfect in-sample fit on a fold - not expected here,
            # but would otherwise blow up the weight to infinity).
            inv_errors = {name: 1.0 / max(mae, 1e-6) for name, mae in cv_mae_by_model.items()}
            total = sum(inv_errors.values())
            self._weights = {name: w / total for name, w in inv_errors.items()}
            # Any model that didn't get a CV MAE (tuning skipped for it,
            # e.g. too few rows) still needs a weight so _average_predict
            # doesn't silently drop it - fall back to a small equal share.
            for name in self._models:
                self._weights.setdefault(name, 1.0 / len(self._models))
            logger.info(
                "Ensemble weights (by CV performance): %s",
                {k: round(v, 3) for k, v in self._weights.items()},
            )
        else:
            self._weights = {name: 1.0 / len(self._models) for name in self._models}

        in_sample_pred = self._average_predict(X)
        self._residuals = np.asarray(y) - in_sample_pred
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._average_predict(X)

    def predict_with_uncertainty(
        self, X: pd.DataFrame, n_iterations: int = 20
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        self._check_fitted()
        point_pred = self._average_predict(X)

        rng = np.random.default_rng(self.random_state)
        # Shape: (n_iterations, n_rows) — one bootstrap draw of residuals
        # per row, per iteration, added on top of the point prediction.
        bootstrap_samples = np.array(
            [
                point_pred + rng.choice(self._residuals, size=len(X), replace=True)
                for _ in range(n_iterations)
            ]
        )

        lower = np.percentile(bootstrap_samples, 2.5, axis=0)
        upper = np.percentile(bootstrap_samples, 97.5, axis=0)
        return point_pred, (lower, upper)

    def save(self, path: str) -> None:
        self._check_fitted()
        joblib.dump(
            {
                "models": self._models,
                "feature_columns": self._feature_columns,
                "residuals": self._residuals,
                "random_state": self.random_state,
                "tuned_params": self.tuned_params_,
                "weights": self._weights,
                "target_lag1_column": self.target_lag1_column,
            },
            path,
        )
        logger.info("EnsembleModel saved to %s", path)

    def load(self, path: str) -> "EnsembleModel":
        bundle = joblib.load(path)
        self._models = bundle["models"]
        self._feature_columns = bundle["feature_columns"]
        self._residuals = bundle["residuals"]
        self.random_state = bundle.get("random_state", self.random_state)
        self.tuned_params_ = bundle.get("tuned_params", {})
        self.target_lag1_column = bundle.get("target_lag1_column")
        self._weights = bundle.get(
            "weights", {name: 1.0 / len(self._models) for name in self._models}
        )
        self._fitted = True
        logger.info("EnsembleModel loaded from %s", path)
        return self

    def _average_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._feature_columns is not None:
            X = X[self._feature_columns]
        preds = np.zeros(len(X))
        for name, model in self._models.items():
            preds += self._weights.get(name, 1.0 / len(self._models)) * model.predict(X)
        return preds

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "EnsembleModel.predict() called before fit()/load() — no trained models available."
            )
