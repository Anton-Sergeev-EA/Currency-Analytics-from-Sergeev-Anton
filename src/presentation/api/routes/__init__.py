from .data import router as data_router
from .forecast import router as forecast_router
from .rag import router as rag_router
from .stats import router as stats_router
from fastapi import APIRouter

router = APIRouter()
router.include_router(data_router, prefix="/data", tags=["data"])
router.include_router(forecast_router, prefix="/forecast", tags=["forecast"])
router.include_router(rag_router, prefix="/rag", tags=["rag"])
router.include_router(stats_router, prefix="/stats", tags=["stats"])
