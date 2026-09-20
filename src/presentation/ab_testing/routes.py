"""
A/B testing endpoints - now backed by src/ab_testing/ab_service.py's real
ABTestService instead of returning fixed mock JSON. See that module's
docstring for what "variant A" and "variant B" actually are.
"""
from fastapi import APIRouter, HTTPException, Query
import logging

from src.ab_testing.ab_service import get_ab_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])


@router.get("/status")
async def get_ab_status():
    """Получить статус A/B-теста."""
    service = get_ab_service()
    return {
        "status": "active",
        "message": "A/B testing is comparing the ML ensemble against a persistence baseline",
        "variants": {"A": "ml_ensemble", "B": "persistence_baseline"},
        "traffic_split": {"A": service.traffic_split, "B": round(1 - service.traffic_split, 4)},
    }


@router.post("/predict")
async def predict_with_ab(
    user_id: str = Query(default=None),
    session_id: str = Query(default=None),
    currency: str = Query(default="usd_rate", description="usd_rate, eur_rate, cny_rate или gbp_rate"),
    days: int = Query(default=1, ge=1, le=30),
):
    """Получить реальный прогноз через A/B-тест (детерминированное назначение варианта)."""
    service = get_ab_service()
    assignment = service.get_variant_for_user(user_id, session_id)

    try:
        result = await service.predict(
            variant=assignment["variant"],
            currency=currency,
            days=days,
            user_id=user_id,
            session_id=session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result["ab_test_active"] = True
    return result


@router.get("/stats")
async def get_ab_stats(days: int = Query(30, ge=1, le=365)):
    """Получить статистику A/B-теста за последние N дней (реальный MAE/MAPE/t-test)."""
    service = get_ab_service()
    return service.get_ab_stats(days=days)


@router.post("/update-ratios")
async def update_traffic_split(split_a: float = Query(..., ge=0, le=1)):
    """Обновить соотношение трафика между вариантами A и B."""
    service = get_ab_service()
    service.traffic_split = split_a
    return {
        "status": "success",
        "new_split": {"A": split_a, "B": round(1 - split_a, 4)},
        "message": "Traffic split updated successfully",
    }


@router.post("/refresh-actuals")
async def refresh_actual_rates():
    """Подтягивает реальные курсы для прогнозов, дата которых уже наступила."""
    service = get_ab_service()
    updated = await service.update_actual_rates()
    return {"status": "success", "updated_records": updated}
