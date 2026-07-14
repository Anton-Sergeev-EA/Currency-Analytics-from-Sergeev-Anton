from fastapi import APIRouter
from src.presentation.api.schemas.response import HealthResponse
from src.infrastructure.data.cache import CacheManager
from src.core.config import settings

router = APIRouter(tags=["Health"])
cache = CacheManager()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    status = "healthy"
    components = {
        "api": "healthy",
        "cache": "healthy" if cache.is_healthy() else "degraded"
    }

    if any(v == "degraded" or v == "unhealthy" for v in components.values()):
        status = "degraded"

    return HealthResponse(
        status=status,
        components=components
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "pong", "version": settings.APP_VERSION}
