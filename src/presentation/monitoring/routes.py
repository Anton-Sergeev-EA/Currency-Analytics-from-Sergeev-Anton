"""
Monitoring dashboard API. Every endpoint here used to return either a
fixed literal or `random.random()`-perturbed numbers labelled "realistic
metrics". None of them do anymore - see ModelEvaluator and the services
each endpoint now actually calls.
"""
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.application.services.data_service import DataService
from src.application.services.model_evaluation import ModelEvaluator
from src.infrastructure.data.cache import CacheManager

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

data_service = DataService()
evaluator = ModelEvaluator()
cache = CacheManager()

MODELS_DIR = "data/models"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    # See presentation/routes.py for why `request` has to be the first
    # positional argument here.
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "cache": "redis" if cache.redis is not None else "in-memory-fallback",
        },
    }


@router.get("/api/models")
async def models_info():
    """Отражает реальное наличие обученных .joblib-файлов на диске."""
    models = {}
    for ccy in ("usd", "eur"):
        path = os.path.join(MODELS_DIR, f"{ccy}_rate_model.joblib")
        loaded = os.path.exists(path)
        models[ccy] = {
            "loaded": loaded,
            "type": "Ensemble (RF, GB, LGB, XGB)" if loaded else "Not trained yet - using statistical trend fallback",
            "path": path,
        }
    return {"models": models, "status": "success"}


@router.get("/api/current-rates")
async def current_rates():
    stats = await data_service.get_stats()
    return {
        "data": {"USD": stats.get("usd_current"), "EUR": stats.get("eur_current")},
        "source": "cbr",
    }


@router.get("/api/model-accuracy")
async def model_accuracy():
    """Реальные метрики walk-forward бэктеста (см. ModelEvaluator), кэшируются на несколько часов."""
    usd_metrics = await evaluator.get_accuracy("usd_rate")
    eur_metrics = await evaluator.get_accuracy("eur_rate")
    return {"data": {"usd": usd_metrics, "eur": eur_metrics}}


@router.get("/api/prediction-test")
async def prediction_test():
    """Реальный прогноз на 1 день от продакшен-модели (или её fallback), а не случайное число."""
    predictions = await evaluator.get_current_predictions()
    return {"predictions": predictions}


@router.get("/api/data-quality")
async def data_quality():
    """Реальная полнота данных: доля дней без пропусков за последние 90 дней."""
    df = await data_service.loader.load_data(90)
    if df is None or df.empty:
        return {"data": {"total_records": 0, "currencies": [], "completeness": 0.0}}

    currencies = [c.replace("_rate", "").upper() for c in ("usd_rate", "eur_rate") if c in df.columns]
    non_null_cols = [c for c in ("usd_rate", "eur_rate") if c in df.columns]
    if non_null_cols:
        completeness = round(float(df[non_null_cols].notna().mean().mean()) * 100, 1)
    else:
        completeness = 0.0

    return {
        "data": {
            "total_records": int(len(df)),
            "currencies": currencies,
            "completeness": completeness,
        }
    }
