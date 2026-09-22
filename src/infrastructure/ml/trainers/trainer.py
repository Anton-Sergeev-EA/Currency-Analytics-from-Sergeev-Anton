import os
import pandas as pd
from typing import Dict
from src.common.logger.logger import get_logger
from src.core.constants import SUPPORTED_CURRENCIES
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
        Обучает и сохраняет ансамблевые модели для всех валют, для которых
        в переданных данных есть колонка (см. src/core/constants.py).
        """
        logger.info("Starting model training process...")
        trained_models = {}

        df_features = self.feature_engineer.create_features(df)
        non_target_cols = ["date"] + SUPPORTED_CURRENCIES
        feature_cols = [c for c in df_features.columns if c not in non_target_cols]

        targets = [c for c in SUPPORTED_CURRENCIES if c in df_features.columns]

        for target in targets:
            logger.info(f"Training ensemble model for target: {target}")

            # Очистка строк с NaN, образовавшимися при генерации лагов
            clean_df = df_features.dropna(subset=feature_cols + [target])

            if len(clean_df) < 30:
                logger.warning(
                    f"Not enough clean rows ({len(clean_df)}) to train a model for {target} - skipping. "
                    "Needs at least a few months of history for the 7-day lag/rolling features to have values."
                )
                continue

            X = clean_df[feature_cols]
            y = clean_df[target]

            # Give the ensemble the option to fall back on a naive
            # "tomorrow = today" candidate for this currency, weighted by
            # its own CV performance just like the tree-based models -
            # see the module docstring in ensemble.py for why.
            lag1_col = f"{target}_lag_1"
            model = EnsembleModel(target_lag1_column=lag1_col if lag1_col in feature_cols else None)
            model.fit(X, y)

            save_path = os.path.join(self.models_dir, f"{target}_model.joblib")
            model.save(save_path)
            logger.info(f"Model for {target} successfully saved to {save_path}")

            trained_models[target] = model

        return trained_models
