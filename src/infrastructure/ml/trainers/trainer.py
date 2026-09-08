import os
import pandas as pd
from typing import Dict
from src.common.logger.logger import get_logger
from src.infrastructure.ml.models.ensemble import EnsembleModel
from src.infrastructure.ml.features.engineer import FeatureEngineer

logger = get_logger(__name__)


class ModelTrainer:
    """
    Класс для подготовки данных, обучения и сохранения ML-моделей.
    """

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.feature_engineer = FeatureEngineer()
        os.makedirs(self.models_dir, exist_ok=True)

    def train_all(self, df: pd.DataFrame) -> Dict[str, EnsembleModel]:
        """
        Обучает и сохраняет ансамблевые модели для всех доступных валютных пар.
        """
        logger.info("Starting model training process...")
        trained_models = {}

        df_features = self.feature_engineer.create_features(df)
        feature_cols = [c for c in df_features.columns if c not in ["date", "usd_rate", "eur_rate"]]

        targets = ["usd_rate", "eur_rate"]

        for target in targets:
            if target not in df_features.columns:
                logger.warning(f"Target column '{target}' not found. Skipping.")
                continue

            logger.info(f"Training ensemble model for target: {target}")

            # Очистка строк с NaN, образовавшимися при генерации лагов
            clean_df = df_features.dropna(subset=feature_cols + [target])

            X = clean_df[feature_cols]
            y = clean_df[target]

            model = EnsembleModel()
            model.fit(X, y)

            save_path = os.path.join(self.models_dir, f"{target}_model.joblib")
            model.save(save_path)
            logger.info(f"Model for {target} successfully saved to {save_path}")

            trained_models[target] = model

        return trained_models
    