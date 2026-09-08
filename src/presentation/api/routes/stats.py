from fastapi import APIRouter
from src.application.services.data_service import DataService

router = APIRouter()
service = DataService()


@router.get("/stats", summary="Получить ключевую статистику")
@router.get("/stats/", include_in_schema=False)
async def get_stats():
    """
    Эндпоинт получения статистики текущих курсов.
    """
    return await service.get_stats()
