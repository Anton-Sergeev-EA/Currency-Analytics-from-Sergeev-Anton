from enum import Enum
from fastapi import APIRouter, Query
from src.application.services.forecast_service import ForecastService

router = APIRouter(tags=["Forecast"])
service = ForecastService()


class Currency(str, Enum):
    ALL = "ALL"
    USD = "usd_rate"
    EUR = "eur_rate"


@router.get("/api/forecast")
async def get_forecast(
    days: int = Query(30, ge=1, le=90, description="Forecast horizon in days"),
    currency: Currency = Query(Currency.ALL, description="Currency to forecast")
):
    """Get currency forecast."""
    data = await service.get_forecast_with_uncertainty(days)
    
    if currency == Currency.ALL:
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
        curr_data = data[currency.value]
        return {
            "currency": currency.value,
            "days": days,
            "mean": curr_data["mean"],
            "lower_bound": curr_data["lower_bound"],
            "upper_bound": curr_data["upper_bound"]
        }
