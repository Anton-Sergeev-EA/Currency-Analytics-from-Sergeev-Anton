from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    type: str = Field(..., description="Response type: forecast, profit, comparison, general")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Прогноз USD:\nТекущий: 75.23 RUB\nЧерез 30 дней: 76.12 RUB",
                "type": "forecast",
                "confidence": 0.8,
                "timestamp": "2026-01-13T12:00:00"
            }
        }


class DataResponse(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Historical data")
    count: int = Field(..., description="Number of records")
    period_days: int = Field(..., description="Period in days")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    status: str = Field(..., description="System status: healthy, degraded, unhealthy")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")
    timestamp: datetime = Field(default_factory=datetime.now)
    