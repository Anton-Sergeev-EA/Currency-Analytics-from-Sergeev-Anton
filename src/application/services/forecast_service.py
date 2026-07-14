from typing import Dict, List, Optional
import pandas as pd
from src.infrastructure.ml.trainers.trainer import ModelTrainer
from src.infrastructure.ml.predictors.predictor import Predictor
from src.infrastructure.data.loader import DataLoader
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)


class ForecastService:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.predictor = Predictor()
        self.loader = DataLoader()
        self.models = None

    def _ensure_models(self):
        if self.models is None:
            logger.info("Models not trained, training now...")
            df = self.loader.load_data(settings.DEFAULT_PERIOD_DAYS)

            if df is not None and len(df) > 30:
                self.models = self.trainer.train_models(df)
                logger.info("Models trained successfully")
            else:
                logger.warning("Not enough data to train models, using fallback")
                self.models = self._create_fallback_models()

    def _create_fallback_models(self) -> dict:
        from src.infrastructure.ml.models.ensemble import EnsembleModel
        import numpy as np

        models = {}
        for currency in ['usd_rate', 'eur_rate']:
            model = EnsembleModel(f"{currency}_ensemble")

            X_fake = pd.DataFrame({
                'feature1': np.random.randn(50),
                'feature2': np.random.randn(50),
                'feature3': np.random.randn(50)
            })
            y_fake = pd.Series(np.random.randn(50) + 75)

            model.train(X_fake, y_fake, cv_folds=2)
            model.is_trained = True
            models[currency] = model
            logger.info(f"Fallback model created for {currency}")

        return models

    def get_forecast(self, days: int = 30) -> Dict[str, List[float]]:
        self._ensure_models()
        df = self.loader.load_data(settings.DEFAULT_PERIOD_DAYS)

        if df is None or len(df) < 10:
            return self._get_simple_forecast(days)

        return self.predictor.predict(self.models, df, days)

    def get_forecast_with_uncertainty(self, days: int = 30) -> Dict[str, Dict]:
        self._ensure_models()
        df = self.loader.load_data(settings.DEFAULT_PERIOD_DAYS)

        if df is None or len(df) < 10:
            return self._get_simple_forecast_with_uncertainty(days)

        return self.predictor.predict_with_uncertainty(self.models, df, days)

    def _get_simple_forecast(self, days: int) -> Dict[str, List[float]]:
        import numpy as np
        df = self.loader.load_data(30)

        if df is None or len(df) == 0:
            base_usd = 75.0
            base_eur = 82.0
        else:
            base_usd = float(df['usd_rate'].iloc[-1])
            base_eur = float(df['eur_rate'].iloc[-1])

        np.random.seed(42)
        usd_forecast = [base_usd + i * 0.05 + np.random.normal(0, 0.2) for i in range(days)]
        eur_forecast = [base_eur + i * 0.04 + np.random.normal(0, 0.18) for i in range(days)]

        return {
            'usd_rate': usd_forecast,
            'eur_rate': eur_forecast
        }

    def _get_simple_forecast_with_uncertainty(self, days: int) -> Dict[str, Dict]:
        forecast = self._get_simple_forecast(days)

        result = {}
        for currency, values in forecast.items():
            std = 0.5
            result[currency] = {
                'mean': values,
                'lower_bound': [v - std for v in values],
                'upper_bound': [v + std for v in values]
            }

        return result
    