import argparse
import asyncio
import sys

from src.application.services.model_evaluation import ModelEvaluator
from src.common.logger.logger import get_logger
from src.core.constants import DEFAULT_TRAINING_WINDOW_DAYS, SUPPORTED_CURRENCIES
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.trainers.trainer import ModelTrainer

logger = get_logger("train_models")


async def main():
    parser = argparse.ArgumentParser(description="Train the currency forecast ensemble models.")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_TRAINING_WINDOW_DAYS,
        help=f"How many days of CBR history to fetch and train on (default: {DEFAULT_TRAINING_WINDOW_DAYS}, ~3 years).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the post-training walk-forward backtest (faster, but you won't see accuracy numbers).",
    )
    args = parser.parse_args()

    print("TRAINING CURRENCY FORECAST MODELS.")
    print(f"\nLoading {args.days} days of data...")

    loader = DataLoader()
    df = await loader.load_data(args.days)

    if df is None or len(df) == 0:
        print("Error: Failed to load data for model training.")
        sys.exit(1)

    print(f"Loaded {len(df)} records ({df['date'].min().date()} .. {df['date'].max().date()}).")
    print(f"  key_rate (CBR macro feature) present: {'yes' if 'key_rate' in df.columns else 'no - fetch failed or 0 rows returned, see WARNING above'}")

    # The loader silently falls back to synthetic demo data if CBR is
    # unreachable (e.g. no internet, or a sandboxed shell with restricted
    # network access) - training "successfully" on that fallback produces
    # models that memorize a random walk, not real exchange-rate behavior.
    # A giveaway: real CBR data has weekend/holiday gaps, demo data is
    # exactly one row per calendar day with no gaps at all.
    calendar_days = (df["date"].max() - df["date"].min()).days + 1
    if len(df) >= calendar_days:
        print(
            "\n*** WARNING: this looks like synthetic DEMO data, not real CBR rates "
            "(one row for every single calendar day, no weekend/holiday gaps). ***\n"
            "*** Check your internet connection - models trained on this will not "
            "reflect real exchange-rate behavior. ***\n"
        )

    print("Starting model training (includes a small hyperparameter search per model - this takes longer than a plain fit)...")
    trainer = ModelTrainer()
    trained = trainer.train_all(df)

    if not trained:
        print("Error: no model could be trained (not enough clean rows after feature engineering).")
        sys.exit(1)

    print(f"\nTrained and saved {len(trained)} model(s): {', '.join(trained.keys())}")

    if not args.skip_eval:
        print("\nRunning walk-forward backtest (held-out last 20 days, per currency)...")
        evaluator = ModelEvaluator()
        for currency in SUPPORTED_CURRENCIES:
            if currency not in trained:
                continue
            metrics = await evaluator.get_accuracy(currency)
            if not metrics.get("available"):
                print(f"  {currency}: backtest unavailable ({metrics.get('reason')})")
                continue
            baseline = metrics["baseline"]
            verdict = "beats naive baseline" if metrics["beats_naive_baseline"] else "DOES NOT beat naive baseline"
            print(
                f"  {currency}: model RMSE={metrics['rmse']} MAE={metrics['mae']} "
                f"MAPE={metrics['mape']}% R2={metrics['r2']}  |  "
                f"naive (yesterday's rate) RMSE={baseline['rmse']} R2={baseline['r2']}  "
                f"-> {verdict}"
            )

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
