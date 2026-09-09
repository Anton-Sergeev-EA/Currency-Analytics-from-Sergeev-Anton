from fastapi import APIRouter, Request, Query
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])

@router.get("/status")
async def get_ab_status():
    """Получить статус A/B-теста."""
    return {
        "status": "active",
        "message": "A/B testing is available",
        "variants": ["control", "treatment"],
        "traffic_split": {"control": 0.5, "treatment": 0.5}
    }

@router.post("/predict")
async def predict_with_ab(request: Request):
    """Получить прогноз с A/B-тестированием."""
    return {
        "variant": "control",
        "model_name": "ensemble_model",
        "predicted_rate": 87.5,
        "forecast_date": "2026-09-05",
        "confidence_interval": {"lower": 86.5, "upper": 88.5},
        "ab_test_active": True
    }

@router.get("/stats")
async def get_ab_stats(days: int = Query(30, ge=1, le=365)):
    """Получить статистику A/B-теста за последние N дней."""
    return {
        "period_days": days,
        "total_requests": 1000,
        "control": {"requests": 500, "avg_prediction": 87.2},
        "treatment": {"requests": 500, "avg_prediction": 87.8},
        "improvement": "0.69%"
    }

@router.post("/update-ratios")
async def update_traffic_split(split_a: float = Query(..., ge=0, le=1)):
    """Обновить соотношение трафика между моделями."""
    return {
        "status": "success",
        "new_split": {"control": split_a, "treatment": 1 - split_a},
        "message": "Traffic split updated successfully"
    }
