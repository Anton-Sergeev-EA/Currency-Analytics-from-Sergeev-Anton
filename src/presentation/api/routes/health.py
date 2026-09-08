from fastapi import APIRouter
from datetime import datetime

from src.presentation.api.schemas.response import HealthResponse
from src.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    # NOTE: no cache layer is wired up yet (there was previously a broken
    # import of a src.infrastructure.data.cache module that does not
    # exist in this codebase — removed). Reporting "not_configured" here
    # is honest about that rather than pretending a cache is being
    # checked.
    components = {
        "api": "healthy",
        "cache": "not_configured",
    }
    status = "healthy"

    return HealthResponse(
        status=status,
        components=components
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "pong", "version": settings.APP_VERSION}


@router.get("/health/verbose")
async def verbose_health_check():
    """Detailed health check with data status."""
    from src.application.services.data_service import DataService
    service = DataService()
    
    try:
        df = await service.get_historical_data(7)
        data_status = {
            "available": len(df) > 0,
            "records": len(df),
            "latest_date": df['date'].max().isoformat() if len(df) > 0 else None,
            "usd_rate": float(df['usd_rate'].iloc[-1]) if len(df) > 0 else None,
            "eur_rate": float(df['eur_rate'].iloc[-1]) if len(df) > 0 else None
        }
    except Exception as e:
        data_status = {"available": False, "error": str(e)}
    
    return {
        "status": "healthy" if data_status.get("available", False) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "data": data_status,
    }

