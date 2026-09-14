from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.application.services.data_service import DataService
from src.application.services.forecast_service import ForecastService
from src.infrastructure.data.cache import CacheManager
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.ml.features.engineer import FeatureEngineer
from src.infrastructure.ml.models.ensemble import EnsembleModel
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

data_service = DataService()
forecast_service = ForecastService()
loader = DataLoader()
feature_engineer = FeatureEngineer()
cache = CacheManager()

MODEL_ACCURACY_CACHE_KEY = "monitoring_model_accuracy_v1"
MODEL_ACCURACY_TTL_SECONDS = 3600  # backtest re-trains models -- too heavy to redo on every 30s dashboard poll

_MODEL_TARGETS = (("usd", "usd_rate"), ("eur", "eur_rate"))


def _model_file_status() -> Dict[str, bool]:
    """Реально ли на диске лежит файл обученной модели для каждой валюты
    (а не просто предположение, что модели загружены)."""
    models_dir = Path(forecast_service.models_dir)
    status = {}
    for key, target in _MODEL_TARGETS:
        model_path = models_dir / f"{target}_model.joblib"
        alt_path = models_dir / f"{key}_model.joblib"
        status[key] = model_path.exists() or alt_path.exists()
    return status


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    # See presentation/routes.py for why `request` has to be the first
    # positional argument here.
    return templates.TemplateResponse(request, "dashboard.html", {"request": request})


@router.get("/api/health")
async def health_check():
    """
    Реальная проверка состояния системы (раньше здесь был жёстко
    прописанный ответ "healthy"/"ok"/"loaded", никогда не отражавший
    ничего реального - фронтенд даже не читал содержимое ответа).

    Проверяем: доступны ли исторические данные (без них прогноз вообще
    не построить), в каком режиме работает кэш (настоящий Redis или
    его in-memory заглушка на этот процесс), и есть ли на диске файлы
    обученных ML-моделей (без них форкаст-сервис тихо переключается на
    статистический трендовый fallback - не авария, но и не то же самое,
    что "модель загружена").
    """
    components: Dict[str, str] = {"api": "ok"}

    if cache.redis is not None and cache.is_healthy():
        components["cache"] = "ok"
    elif cache.is_healthy():
        components["cache"] = "degraded"  # работает, но через in-memory fallback без Redis
    else:
        components["cache"] = "error"

    try:
        df = await loader.load_data()
        components["data"] = "ok" if df is not None and not df.empty else "error"
    except Exception as exc:
        logger.warning(f"Health check: data source unavailable: {exc}")
        components["data"] = "error"

    model_status = _model_file_status()
    components["models"] = "ok" if all(model_status.values()) else "degraded"

    if components["data"] == "error":
        overall = "unhealthy"
    elif "error" in components.values() or "degraded" in components.values():
        overall = "degraded"
    else:
        overall = "healthy"

    return {"status": overall, "components": components}


@router.get("/api/models")
async def models_info():
    """Реальный статус ML-моделей: файл на диске либо есть, либо нет
    (раньше это был статический словарь с loaded=True для обеих валют
    независимо от того, существуют ли файлы .joblib)."""
    model_status = _model_file_status()
    models = {
        key: {
            "loaded": loaded,
            "type": "Ensemble (RF, GB, LGB, XGB, Ridge)"
            if loaded
            else "модель не найдена - используется статистический трендовый fallback",
        }
        for key, loaded in model_status.items()
    }
    return {"models": models, "status": "success"}


@router.get("/api/current-rates")
async def current_rates():
    """Реальные текущие курсы (раньше здесь была захардкоженная заглушка)."""
    stats = await data_service.get_stats()
    return {
        "data": {"USD": stats["usd_current"], "EUR": stats["eur_current"]},
        "source": "cbr"
    }


@router.get("/api/model-accuracy")
async def model_accuracy():
    """
    Реальные метрики качества моделей (RMSE/MAE/MAPE/R²), посчитанные на
    отложенной выборке (train/test split по времени) -- раньше эти числа
    генерировались случайно (`random.random()`) и не отражали ничего
    реального.

    Обучение на полном наборе признаков не быстрое, а страница опрашивает
    этот эндпоинт каждые 30 секунд, поэтому результат кешируется на час:
    метрики моделей не меняются from request to request, только когда
    появляются новые исторические данные.
    """
    cached = cache.get(MODEL_ACCURACY_CACHE_KEY)
    if cached is not None:
        return {"data": cached}

    data = await _compute_real_model_accuracy()
    cache.set(MODEL_ACCURACY_CACHE_KEY, data, MODEL_ACCURACY_TTL_SECONDS)
    return {"data": data}


async def _compute_real_model_accuracy() -> Dict[str, Any]:
    df = await loader.load_data(365)
    if df is None or df.empty:
        return {}

    df_features = feature_engineer.create_features(df)
    feature_cols = [c for c in df_features.columns if c not in ["date", "usd_rate", "eur_rate"]]

    results: Dict[str, Any] = {}
    for target in ["usd_rate", "eur_rate"]:
        if target not in df_features.columns:
            continue

        clean_df = df_features.dropna(subset=feature_cols + [target])
        # Отложенная выборка по времени (последние ~20%, не менее 5 записей) -
        # модель обучается только на данных ДО этого окна, иначе метрики
        # были бы "внутривыборочными" и слишком оптимистичными.
        test_size = max(int(len(clean_df) * 0.2), 5)
        if len(clean_df) < test_size + 10:
            logger.warning(f"Not enough data to backtest {target} (have {len(clean_df)} rows). Skipping.")
            continue

        train_df = clean_df.iloc[:-test_size]
        test_df = clean_df.iloc[-test_size:]

        model = EnsembleModel()
        model.fit(train_df[feature_cols], train_df[target])
        y_pred = model.predict(test_df[feature_cols])
        y_true = test_df[target].to_numpy()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        r2 = float(r2_score(y_true, y_pred))

        key = target.replace("_rate", "")
        results[key] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
        }

    return results


@router.get("/api/prediction-test")
async def prediction_test():
    """Реальный прогноз на 1 день вперёд (раньше -- случайные числа вокруг
    нереалистичных базовых значений, включая несуществующую в проекте валюту GBP)."""
    usd_forecast = await forecast_service.get_forecast(days=1, currency="usd")
    eur_forecast = await forecast_service.get_forecast(days=1, currency="eur")

    predictions = {}
    if isinstance(usd_forecast, list) and usd_forecast:
        predictions["usd"] = usd_forecast[0]["forecast"]
    if isinstance(eur_forecast, list) and eur_forecast:
        predictions["eur"] = eur_forecast[0]["forecast"]

    return {"predictions": predictions}


@router.get("/api/data-quality")
async def data_quality():
    """
    Реальное качество данных: сколько записей реально есть и какая доля
    ожидаемых (рабочих) дней в периоде покрыта данными -- раньше
    "полнота" была случайным числом около 99%, никак не связанным с
    фактическими данными.
    """
    df = await loader.load_data(365)
    if df is None or df.empty:
        return {"data": {"total_records": 0, "currencies": ["USD", "EUR"], "completeness": 0.0}}

    completeness = 100.0
    if "date" in df.columns and len(df) > 1:
        date_min, date_max = df["date"].min(), df["date"].max()
        expected_days = len(pd.bdate_range(date_min, date_max))
        if expected_days > 0:
            completeness = min(100.0, round(len(df) / expected_days * 100, 1))

    return {
        "data": {
            "total_records": len(df),
            "currencies": ["USD", "EUR"],
            "completeness": completeness
        }
    }
