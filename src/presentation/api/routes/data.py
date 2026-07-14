from fastapi import APIRouter, Query
from src.application.services.data_service import DataService
from src.presentation.api.schemas.response import DataResponse

router = APIRouter(tags=["Data"])
service = DataService()


@router.get("/api/data", response_model=DataResponse)
async def get_data(
        period_days: int = Query(90, ge=1, le=730, description="Number of days of historical data"),
        refresh: bool = Query(False, description="Force refresh cache")
):
    """Get historical currency data."""
    df = service.get_historical_data(period_days, refresh)

    return DataResponse(
        data=df.to_dict('records'),
        count=len(df),
        period_days=period_days
    )
