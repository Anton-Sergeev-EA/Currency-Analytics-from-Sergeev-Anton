"""
EnsembleModel — combines LightGBM, XGBoost, Random Forest, and Gradient
Boosting regressors into a single averaged predictor with bootstrap-based
uncertainty estimates.

This class was referenced (imported and called) by `ModelTrainer`,
`ForecastService`, and the A/B testing service, but did not exist
anywhere in the codebase or its git history — reconstructed here from the
exact interface those three call sites already depend on:

    model = EnsembleModel()
    model.fit(X, y)
    model.predict(X) -> array of length len(X)
    model.predict_with_uncertainty(X, n_iterations=20) -> (mean, (lower, upper))
    model.save(path)
    EnsembleModel().load(path)

## Design notes

**Point prediction** is the simple mean of the four base regressors'
predictions — a standard, low-variance ensembling approach that doesn't
require the base learners to be correlated in any particular way.

**Uncertainty estimation** uses residual bootstrap rather than refitting
models at prediction time (refitting 4 models x N iterations on every
forecast request would be far too slow for an API endpoint). During
`fit()`, in-sample residuals (true - predicted) are computed once and
stored. At prediction time, `n_iterations` bootstrap samples are drawn
from those residuals and added to the point prediction; the 2.5th and
97.5th percentiles of the resulting distribution form the (lower, upper)
bound — a widely-used, cheap approximation of a ~95% prediction interval.
It is a simplification (in-sample residuals tend to slightly
underestimate true out-of-sample uncertainty), documented here rather
than hidden, so it can be swapped for out-of-fold residuals later without
changing any caller.

**Hyperparameter tuning**: each base model used to fit with plain library
defaults, which for ~500 rows and 30+ correlated lag/rolling features is
a recipe for overfitting the training set and doing no better than noise
out of sample (the walk-forward backtest in model_evaluation.py first
surfaced this as a negative R2). `fit()` now runs a small, bounded
`RandomizedSearchCV` per base model with `TimeSeriesSplit` folds instead
of a random/shuffled split — shuffling would let a fold "predict the
past from the future", which is not a real forecasting scenario. The
search space and iteration count are kept deliberately small (a handful
of candidates x a few folds) so this stays fast and memory-light enough
for a small VDS; it is not an exhaustive grid search.
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


class EnsembleModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._models: dict[str, object] = {
            "lightgbm": LGBMRegressor(random_state=random_state, verbosity=-1),
            "xgboost": XGBRegressor(random_state=random_state, verbosity=0),
            "random_forest": RandomForestRegressor(random_state=random_state, n_jobs=1),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }
        self._feature_columns: Optional[list[str]] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False
        self.tuned_params_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = True) -> "EnsembleModel":
        logger.info("Fitting ensemble (%d base models) on %d rows", len(self._models), len(X))
        self._feature_columns = list(X.columns)

        can_tune = tune and len(X) >= _MIN_ROWS_TO_TUNE
        n_splits = min(_TUNE_CV_SPLITS, max(2, len(X) // 30)) if can_tune else 0

        for name, model in self._models.items():
            if can_tune:
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
                logger.info(
                    "Tuned %s: %s (CV MAE=%.4f)", name, search.best_params_, -search.best_score_
                )
            else:
                model.fit(X, y)

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
        self._fitted = True
        logger.info("EnsembleModel loaded from %s", path)
        return self

    def _average_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._feature_columns is not None:
            X = X[self._feature_columns]
        predictions = np.column_stack([model.predict(X) for model in self._models.values()])
        return predictions.mean(axis=1)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "EnsembleModel.predict() called before fit()/load() — no trained models available."
            )
