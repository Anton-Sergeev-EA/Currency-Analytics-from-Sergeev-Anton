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
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class EnsembleModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._models: dict[str, object] = {
            "lightgbm": LGBMRegressor(random_state=random_state, verbosity=-1),
            "xgboost": XGBRegressor(random_state=random_state, verbosity=0),
            "random_forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }
        self._feature_columns: Optional[list[str]] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        logger.info("Fitting ensemble (%d base models) on %d rows", len(self._models), len(X))
        self._feature_columns = list(X.columns)

        for name, model in self._models.items():
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
