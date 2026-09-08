from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
import json
import logging

from ...ab_testing.ab_service import get_ab_service, ABTestService
from ...config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])

class PredictionRequest(BaseModel):
    currency_pair: str = Field(default="USD/RUB", description="Валютная пара")
    forecast_days: int = Field(default=3, ge=1, le=30, description="Горизонт прогноза")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")

class PredictionResponse(BaseModel):
    variant: str
    model_name: str
    predicted_rate: float
    forecast_date: datetime
    confidence_interval: Optional[dict] = None
    ab_test_active: bool = True

@router.get("/status")
async def get_ab_status(service: ABTestService = Depends(get_ab_service)):
    """Получить статус A/B-теста."""
    return {
        "status": "active",
        "traffic_split": service.traffic_split,
        "models": {
            "A": service.variant_mapping['A']['name'],
            "B": service.variant_mapping['B']['name']
        }
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict_with_ab(
    request: Request,
    pred_request: PredictionRequest,
    service: ABTestService = Depends(get_ab_service)
):
    """
    Получить прогноз с A/B-тестированием.
    Пользователи распределяются по группам случайным образом.
    """
    try:
        # Определяем вариант для пользователя.
        user_id = pred_request.user_id or request.client.host
        session_id = pred_request.session_id or str(request.headers.get('X-Session-ID', ''))
        
        variant_info = service.get_variant_for_user(user_id, session_id)
        
        # Получаем прогноз от модели.
        model = variant_info['model']
        # Здесь должна быть ваша логика получения прогноза.
        # Пример:
        forecast_date = datetime.now() + timedelta(days=pred_request.forecast_days)
        predicted_rate = model.predict(currency_pair=pred_request.currency_pair, 
                                       days=pred_request.forecast_days)
        
        # Логируем прогноз.
        service.log_prediction(
            user_id=user_id,
            session_id=session_id,
            variant=variant_info['variant'],
            predicted_rate=predicted_rate,
            forecast_date=forecast_date
        )
        
        return PredictionResponse(
            variant=variant_info['variant'],
            model_name=variant_info['model_name'],
            predicted_rate=predicted_rate,
            forecast_date=forecast_date,
            confidence_interval={"lower": predicted_rate * 0.98, "upper": predicted_rate * 1.02}
        )
        
    except Exception as e:
        logger.error(f"Error in A/B prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_ab_stats(
    days: int = 30,
    service: ABTestService = Depends(get_ab_service)
):
    """Получить статистику A/B-теста за последние N дней."""
    stats = service.get_ab_stats(days)
    
    # Добавляем рекомендацию на основе статистики.
    if 'statistical_significance' in stats:
        sig = stats['statistical_significance']
        if sig['is_significant']:
            # Определяем, какая модель лучше.
            variant_a_mae = stats.get('variant_A', {}).get('mae', float('inf'))
            variant_b_mae = stats.get('variant_B', {}).get('mae', float('inf'))
            
            if variant_a_mae < variant_b_mae:
                stats['recommendation'] = "Модель A показывает статистически значимо лучшие результаты. Рекомендуется сделать её основной."
            else:
                stats['recommendation'] = "Модель B показывает статистически значимо лучшие результаты. Рекомендуется сделать её основной."
        else:
            stats['recommendation'] = "Статистически значимых различий не обнаружено. Продолжайте тест или соберите больше данных."
    
    return stats

@router.post("/update-ratios")
async def update_traffic_split(
    split_a: float,
    service: ABTestService = Depends(get_ab_service)
):
    """Обновить соотношение трафика между моделями (только для админов)."""
    if not 0 <= split_a <= 1:
        raise HTTPException(status_code=400, detail="Split must be between 0 and 1")
    
    service.traffic_split = split_a
    return {
        "message": f"Traffic split updated to A={split_a:.0%}, B={1-split_a:.0%}"
    }
