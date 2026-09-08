from fastapi import APIRouter, Query
from src.application.services.forecast_service import ForecastService

router = APIRouter()
service = ForecastService()


@router.get("/forecast", summary="Получить прогноз курсов валют")
@router.get("/forecast/", include_in_schema=False)
async def get_forecast(
    days: int = Query(default=7, ge=1, le=30),
    currency: str = Query(default="ALL")
):
    """
    Эндпоинт получения 7-дневного прогноза курсов валют.
    """
    return await service.get_forecast(days=days, currency=currency)
