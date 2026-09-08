from fastapi import APIRouter, BackgroundTasks
from src.infrastructure.data.cache import CacheManager
from src.infrastructure.data.loader import DataLoader
from src.application.services.data_service import DataService
import logging

router = APIRouter(tags=["Admin"])
logger = logging.getLogger(__name__)
cache = CacheManager()
data_service = DataService()


@router.post("/api/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh data and clear cache."""
    logger.info("Data refresh triggered")

    cache.clear()

    # Preload fresh data.
    background_tasks.add_task(
        data_service.get_historical_data,
        period_days=90,
        refresh=True
    )

    return {
        "status": "success",
        "message": "Data refresh started in background"
    }


@router.get("/api/cache/status")
async def cache_status():
    """Check cache status."""
    return {
        "redis_available": cache.redis is not None,
        "local_cache_size": len(cache.local_cache)
    }


@router.post("/api/force-refresh")
async def force_refresh():
    """Force refresh all data."""
    try:
        loader = DataLoader()
        df = await loader.refresh_data(365)
        
        # Проверка на пустые данные
        if df is None or len(df) == 0:
            logger.error("Force refresh completed, but NO data was loaded!")
            return {
                "status": "error",
                "message": "Refresh completed, but no data could be loaded from the source. Please check the CBR API or internet connection."
            }
        
        return {
            "status": "success",
            "message": f"Data refreshed successfully. Loaded {len(df)} records.",
            "last_date": df['date'].max().isoformat() if len(df) > 0 else None
        }
    except Exception as e:
        logger.error(f"Force refresh failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
