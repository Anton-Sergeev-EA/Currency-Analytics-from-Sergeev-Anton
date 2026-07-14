import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.trainers.trainer import ModelTrainer
from src.infrastructure.ml.features.engineer import prepare_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("TRAINING CURRENCY FORECAST MODELS.")

    print("\nLoading data")
    loader = DataLoader()
    df = loader.load_data(365)

    if df is None or len(df) == 0:
        print("ERROR: No data loaded! Using demo data...")
        df = loader._generate_demo_data(365)

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"USD: {df['usd_rate'].iloc[-1]:.2f} RUB")
    print(f"EUR: {df['eur_rate'].iloc[-1]:.2f} RUB")

    print("\n2. Training models")
    trainer = ModelTrainer()
    models = trainer.train_models(df)

    if models:
        print(f"Trained models: {list(models.keys())}")
        print("Models saved to: data/models/")
    else:
        print("No models trained!")
        return

    print("\n3. Testing predictions")
    from src.infrastructure.ml.predictors.predictor import Predictor
    predictor = Predictor()

    try:
        processed = prepare_features(df)
        forecast = predictor.predict(models, processed, days=7)

        if 'usd_rate' in forecast:
            print(f"USD 7-day forecast: {[round(x, 2) for x in forecast['usd_rate'][:5]]}...")
        if 'eur_rate' in forecast:
            print(f"EUR 7-day forecast: {[round(x, 2) for x in forecast['eur_rate'][:5]]}...")

        print("MODELS TRAINED SUCCESSFULLY.")
        print("\nNow restart the server:")
        print("python -m src.main")
        print("\nThen test forecast in Swagger UI:")
        print("http://localhost:8000/api/docs")

    except Exception as e:
        print(f"Prediction test failed: {e}")


if __name__ == "__main__":
    main()
