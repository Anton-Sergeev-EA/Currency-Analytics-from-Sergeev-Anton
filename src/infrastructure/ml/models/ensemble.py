"""
EnsembleModel — combines LightGBM, XGBoost, Random Forest, Gradient
Boosting, and Ridge regression into a single, performance-weighted
predictor with bootstrap-based uncertainty estimates.

This class was referenced (imported and called) by `ModelTrainer`,
`ForecastService`, and the A/B testing service, but did not exist
anywhere in the codebase or its git history — reconstructed here from
the exact interface those three call sites already depend on:

    model = EnsembleModel()
    model.fit(X, y)
    model.predict(X) -> array of length len(X)
    model.predict_with_uncertainty(X, n_iterations=20) -> (mean, (lower, upper))
    model.save(path)
    EnsembleModel().load(path)

## Design notes

**Point prediction** is a weighted average of the base regressors'
predictions. It started as a plain, equal-weight mean (the textbook
low-variance ensembling baseline), but a real backtest against CBR
exchange-rate data showed that badly underperforming on a currency pair
that has been on a sustained multi-month trend: RandomForest, Gradient
Boosting, LightGBM and XGBoost are all decision-tree ensembles, and a
tree can only predict values it saw during training (a leaf node's
output is bounded by the training targets that landed in it) -- it
cannot extrapolate beyond the training range. When the most recent
~20% of a steadily rising series is held out as the test set (exactly
what the /monitoring/api/model-accuracy backtest does), every tree in
the ensemble is systematically forced to under-predict, and R² came out
negative or barely positive.

A `Ridge` regressor (L2-regularized linear regression, `StandardScaler`
first since Ridge's penalty is scale-sensitive) doesn't have this
limitation -- a linear model extrapolates by construction. Added it as
a fifth ensemble member, and switched from an equal-weight mean to
**inverse-MSE weighting**: during `fit()`, the last ~15% of the
training data (chronologically -- never the model-accuracy endpoint's
own held-out test set, which stays untouched) is used purely to score
each base model's validation MSE, and each model's share of the final
average is proportional to `1 / MSE`. A model that is twice as
accurate (half the MSE) gets twice the vote. This is a standard,
named technique (inverse-variance/precision weighting) rather than an
arbitrarily tuned knob -- and it consistently gives Ridge most of the
weight on trending series while leaving the tree models free to pull
weight back on choppier, mean-reverting stretches where they do better,
so the ensemble adapts rather than hard-switching to "just use Ridge".
After the weights are fixed, every model is refit on the *full*
training set (the validation slice was only used to score them) so no
training data is wasted on the final deployed model.

**Uncertainty estimation** uses residual bootstrap rather than refitting
models at prediction time (refitting 5 models x N iterations on every
forecast request would be far too slow for an API endpoint). During
`fit()`, in-sample residuals (true - weighted-predicted) are computed
once and stored. At prediction time, `n_iterations` bootstrap samples
are drawn from those residuals and added to the point prediction; the
2.5th and 97.5th percentiles of the resulting distribution form the
(lower, upper) bound — a widely-used, cheap approximation of a ~95%
prediction interval. It is a simplification (in-sample residuals tend
to slightly underestimate true out-of-sample uncertainty), documented
here rather than hidden, so it can be swapped for out-of-fold residuals
later without changing any caller.
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Доля обучающих данных (по хронологии, последние N%), отводимая ВНУТРИ
# fit() только для оценки веса каждой базовой модели. Не пересекается с
# тестовой выборкой бэктеста /monitoring/api/model-accuracy -- та видит
# уже итоговую, обученную на всех данных модель.
_VALIDATION_FRACTION = 0.15
_MIN_VALIDATION_ROWS = 3
_MIN_TRAIN_ROWS_FOR_WEIGHTING = 10  # меньше -- веса ненадёжны, используем равные


class EnsembleModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._models: dict[str, object] = {
            "lightgbm": LGBMRegressor(random_state=random_state, verbosity=-1),
            "xgboost": XGBRegressor(random_state=random_state, verbosity=0),
            "random_forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state)),
        }
        self._weights: dict[str, float] = {name: 1.0 / len(self._models) for name in self._models}
        self._feature_columns: Optional[list[str]] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        logger.info("Fitting ensemble (%d base models) on %d rows", len(self._models), len(X))
        self._feature_columns = list(X.columns)

        self._weights = self._compute_validation_weights(X, y)
        logger.info(
            "Ensemble weights from internal validation split: %s",
            {name: round(w, 3) for name, w in self._weights.items()},
        )

        for name, model in self._models.items():
            model.fit(X, y)

        in_sample_pred = self._average_predict(X)
        self._residuals = np.asarray(y) - in_sample_pred
        self._fitted = True
        return self

    def _compute_validation_weights(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Оценивает MSE каждой базовой модели на последних ~15% обучающих
        данных (обучаясь только на предшествующей им части) и возвращает
        веса, пропорциональные 1/MSE -- модель с вдвое меньшей ошибкой
        получает вдвое больший голос в финальном усреднении.

        При слишком малом объёме данных веса ненадёжны -- возвращает
        равные веса, как и было до появления этого механизма.
        """
        equal_weights = {name: 1.0 / len(self._models) for name in self._models}

        n = len(X)
        val_size = max(int(n * _VALIDATION_FRACTION), _MIN_VALIDATION_ROWS)
        if n - val_size < _MIN_TRAIN_ROWS_FOR_WEIGHTING:
            logger.warning(
                "Only %d rows available -- too few to score ensemble weights reliably; using equal weights", n
            )
            return equal_weights

        X_train, y_train = X.iloc[:-val_size], y.iloc[:-val_size]
        X_val, y_val = X.iloc[-val_size:], y.iloc[-val_size:]
        y_val_arr = np.asarray(y_val)

        mse_by_model: dict[str, float] = {}
        for name, model in self._models.items():
            # Клонируем через сохранение/восстановление параметров не нужно --
            # sklearn/lightgbm/xgboost эстиматоры можно просто переобучить,
            # финальный fit() ниже всё равно переобучит их на всех данных.
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            mse_by_model[name] = float(np.mean((y_val_arr - val_pred) ** 2))

        inv_mse = {name: 1.0 / max(mse, 1e-9) for name, mse in mse_by_model.items()}
        total = sum(inv_mse.values())
        return {name: w / total for name, w in inv_mse.items()}

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
                "weights": self._weights,
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
        # `weights` didn't exist in model files saved before this change --
        # fall back to equal weights rather than failing to load them.
        self._weights = bundle.get("weights") or {name: 1.0 / len(self._models) for name in self._models}
        self._feature_columns = bundle["feature_columns"]
        self._residuals = bundle["residuals"]
        self.random_state = bundle.get("random_state", self.random_state)
        self._fitted = True
        logger.info("EnsembleModel loaded from %s", path)
        return self

    def _average_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._feature_columns is not None:
            X = X[self._feature_columns]
        weighted_sum = None
        for name, model in self._models.items():
            contribution = model.predict(X) * self._weights.get(name, 0.0)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        return weighted_sum

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "EnsembleModel.predict() called before fit()/load() — no trained models available."
            )
