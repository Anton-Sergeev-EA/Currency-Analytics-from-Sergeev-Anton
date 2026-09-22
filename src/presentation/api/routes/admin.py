from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from src.core.config import settings
from src.infrastructure.data.cache import CacheManager
from src.infrastructure.data.loader import DataLoader
from src.application.services.data_service import DataService
import logging

logger = logging.getLogger(__name__)


def require_admin_key(x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key")) -> None:
    """Every route below used to be reachable by anyone on the public
    internet with no credential at all - a real finding from a security
    review: POST /api/force-refresh triggers genuine work (a live fetch
    from the Bank of Russia plus a full data reload), and none of these
    three endpoints had any rate limiting either, so they were open to
    trivial abuse by anyone who found the URL (it's in this repo's own
    README). Reuses SECRET_KEY (declared in config.py since day one but
    never actually checked anywhere, until now) rather than introducing a
    separate setting nobody would remember to set.

    If SECRET_KEY is still the default placeholder, this still enforces
    the header check - but that "secret" is the same one printed in this
    public repo's .env.example, so it isn't really protecting anything.
    main.py's startup logging warns about exactly that case; this
    function doesn't special-case it, both to keep the check itself
    simple and because silently allowing every request through in that
    case would make a not-yet-configured deployment look secure when it
    isn't.
    """
    if not x_admin_key or x_admin_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Admin-Key header",
        )


router = APIRouter(tags=["Admin"], dependencies=[Depends(require_admin_key)])
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
