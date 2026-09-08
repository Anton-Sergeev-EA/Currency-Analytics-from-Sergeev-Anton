import pandas as pd
import numpy as np
from datetime import timedelta
from src.infrastructure.ml.features.engineer import prepare_features
import logging

logger = logging.getLogger(__name__)


class Predictor:
    def predict(self, models: dict, df: pd.DataFrame, days: int = 30) -> dict:
        logger.info(f"Generating forecasts for {days} days")
        results = {}

        for currency, model in models.items():
            predictions = []
            current_df = df.copy()

            for step in range(days):
                try:
                    processed = prepare_features(current_df)
                    features = processed.drop(columns=['date', currency])

                    features = features.replace([np.inf, -np.inf], 0)
                    features = features.fillna(0)
                    features = features.clip(-1e6, 1e6)

                    pred = model.predict(features.iloc[-1:])[0]

                    pred = np.clip(pred, 30, 200)
                    predictions.append(pred)

                    next_date = current_df['date'].iloc[-1] + timedelta(days=1)
                    new_row = current_df.iloc[-1].copy()
                    new_row['date'] = next_date
                    new_row[currency] = pred
                    current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

                except Exception as e:
                    logger.warning(f"Prediction error at step {step}: {e}")
                    if predictions:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(float(current_df[currency].iloc[-1]))

            results[currency] = predictions

        return results

    def predict_with_uncertainty(self, models: dict, df: pd.DataFrame, days: int = 30) -> dict:
        logger.info(f"Generating forecasts with uncertainty for {days} days")
        results = {}

        for currency, model in models.items():
            means = []
            lower_bounds = []
            upper_bounds = []
            current_df = df.copy()

            for step in range(days):
                try:
                    processed = prepare_features(current_df)
                    features = processed.drop(columns=['date', currency])

                    features = features.replace([np.inf, -np.inf], 0)
                    features = features.fillna(0)
                    features = features.clip(-1e6, 1e6)

                    mean, (lower, upper) = model.predict_with_uncertainty(
                        features.iloc[-1:],
                        n_iterations=20
                    )

                    mean = np.clip(mean, 30, 200)[0]
                    lower = np.clip(lower, 30, 200)[0]
                    upper = np.clip(upper, 30, 200)[0]

                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)

                    next_date = current_df['date'].iloc[-1] + timedelta(days=1)
                    new_row = current_df.iloc[-1].copy()
                    new_row['date'] = next_date
                    new_row[currency] = mean
                    current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

                except Exception as e:
                    logger.warning(f"Uncertainty error at step {step}: {e}")
                    if means:
                        means.append(means[-1])
                        lower_bounds.append(lower_bounds[-1])
                        upper_bounds.append(upper_bounds[-1])
                    else:
                        last_val = float(current_df[currency].iloc[-1])
                        means.append(last_val)
                        lower_bounds.append(last_val * 0.97)
                        upper_bounds.append(last_val * 1.03)

            results[currency] = {
                'mean': means,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            }

        return results
