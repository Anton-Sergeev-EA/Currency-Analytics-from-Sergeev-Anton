from fastapi import APIRouter, Depends, HTTPException
from src.application.services.data_service import DataService

router = APIRouter()


def get_data_service() -> DataService:
    return DataService()


@router.get("/api/data")
def get_data(
    period_days: int = 180, 
    refresh: bool = False, 
    service: DataService = Depends(get_data_service)
):
    try:
        df = service.get_historical_data(period_days, refresh)
        return {
            "status": "success",
            "count": len(df),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
