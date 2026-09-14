from .data import router as data_router
from .forecast import router as forecast_router
from .rag import router as rag_router
from .stats import router as stats_router
from fastapi import APIRouter

# NOTE: no prefix= here -- each sub-router below already declares its own
# full path (e.g. data.py uses @router.get("/data")). Adding a prefix on
# top of that used to double the path segment (/api/data/data instead of
# /api/data).
router = APIRouter()
router.include_router(data_router, tags=["data"])
router.include_router(forecast_router, tags=["forecast"])
router.include_router(rag_router, tags=["rag"])
router.include_router(stats_router, tags=["stats"])
