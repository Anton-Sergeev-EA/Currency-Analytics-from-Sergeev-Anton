from fastapi import APIRouter, Query
from src.application.services.forecast_service import ForecastService
from src.core.constants import Currency

router = APIRouter(tags=["Forecast"])
service = ForecastService()

@router.get("/api/forecast")
async def get_forecast(
    days: int = Query(30, ge=1, le=90, description="Forecast horizon in days"),
    currency: Currency = Query(Currency.ALL, description="Currency to forecast")
):
    """Get currency forecast."""
    if currency == Currency.ALL:
        data = service.get_forecast_with_uncertainty(days)
        return {
            "currency": "ALL",
            "days": days,
            "mean": {
                "usd_rate": data["usd_rate"]["mean"],
                "eur_rate": data["eur_rate"]["mean"]
            },
            "lower_bound": {
                "usd_rate": data["usd_rate"]["lower_bound"],
                "eur_rate": data["eur_rate"]["lower_bound"]
            },
            "upper_bound": {
                "usd_rate": data["usd_rate"]["upper_bound"],
                "eur_rate": data["eur_rate"]["upper_bound"]
            }
        }
    else:
        data = service.get_forecast_with_uncertainty(days)[currency.value]
        return {
            "currency": currency.value,
            "days": days,
            "mean": data["mean"],
            "lower_bound": data["lower_bound"],
            "upper_bound": data["upper_bound"]
        }
    