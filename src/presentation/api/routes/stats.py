from fastapi import APIRouter
from src.application.services.data_service import DataService
from src.core.config import settings

router = APIRouter(tags=["Statistics"])
service = DataService()


@router.get("/api/stats")
async def get_stats():
    """Get statistics about currency data."""
    df = service.get_historical_data(settings.DEFAULT_PERIOD_DAYS)

    if len(df) == 0:
        return {"error": "No data available"}

    return {
        "total_days": len(df),
        "date_range": {
            "start": df['date'].min().isoformat(),
            "end": df['date'].max().isoformat()
        },
        "usd": {
            "current": float(df['usd_rate'].iloc[-1]),
            "mean": float(df['usd_rate'].mean()),
            "std": float(df['usd_rate'].std()),
            "min": float(df['usd_rate'].min()),
            "max": float(df['usd_rate'].max())
        },
        "eur": {
            "current": float(df['eur_rate'].iloc[-1]),
            "mean": float(df['eur_rate'].mean()),
            "std": float(df['eur_rate'].std()),
            "min": float(df['eur_rate'].min()),
            "max": float(df['eur_rate'].max())
        }
    }
