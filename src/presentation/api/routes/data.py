from fastapi import APIRouter, Query
from src.application.services.data_service import DataService

router = APIRouter()
service = DataService()


@router.get("/data", summary="Получить исторические данные")
@router.get("/data/", include_in_schema=False)
async def get_data(period_days: int = Query(default=180, ge=1, le=365)):
    """
    Эндпоинт получения исторических курсов валют.
    """
    return await service.get_historical_data(period_days)
