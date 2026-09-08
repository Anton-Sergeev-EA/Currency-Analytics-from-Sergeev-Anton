import asyncio
import sys
from src.common.logger.logger import get_logger
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.trainers.trainer import ModelTrainer

logger = get_logger("train_models")


async def main():
    print("TRAINING CURRENCY FORECAST MODELS.")
    print("\nLoading data...")

    loader = DataLoader()
    df = await loader.load_data()

    if df is None or len(df) == 0:
        print("Error: Failed to load data for model training.")
        sys.exit(1)

    print(f"Loaded {len(df)} records. Starting model training...")

    trainer = ModelTrainer()
    trainer.train_all(df)

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
