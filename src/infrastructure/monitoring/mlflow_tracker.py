"""
Тонкая, отказоустойчивая обёртка над MLflow tracking client.

Логирование сюда не должно ронять и не должно замедлять основной путь
запроса: бэктест точности (/monitoring/api/model-accuracy) кешируется
на час и всё равно должен вернуть результат пользователю, даже если
сервер MLflow сейчас недоступен или отвечает медленно, а обучение
моделей (train_models.py) не должно падать целиком из-за того, что не
получилось залогировать один run. Поэтому любая ошибка здесь только
предупреждается в лог и проглатывается - вызывающий код продолжает
работать как ни в чём не бывало.

Использует `mlflow-skinny` (только tracking-клиент, без веб-сервера и
его зависимостей) - сервер MLflow уже отдельно поднят на VDS.
"""
import logging
from typing import Any, Dict, Optional

from src.core.config import settings

logger = logging.getLogger(__name__)

_mlflow_module: Optional[Any] = None
_tracking_uri_set = False


def _get_mlflow() -> Optional[Any]:
    """Лениво импортирует и настраивает mlflow, только если он реально
    нужен (USE_MLFLOW=True) - в остальных случаях (тесты, локальная
    разработка без сервера) даже не пытается его импортировать."""
    global _mlflow_module, _tracking_uri_set

    if not settings.USE_MLFLOW:
        return None

    if _mlflow_module is None:
        try:
            import mlflow  # noqa: PLC0415 - осознанно ленивый импорт, см. докстринг модуля
            _mlflow_module = mlflow
        except Exception as exc:
            logger.warning(f"MLflow client недоступен (пакет не установлен?): {exc}")
            return None

    if not _tracking_uri_set:
        try:
            _mlflow_module.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            _tracking_uri_set = True
        except Exception as exc:
            logger.warning(f"Не удалось задать MLFLOW_TRACKING_URI: {exc}")
            return None

    return _mlflow_module


def log_run(
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Optional[float]]] = None,
    artifact_path: Optional[str] = None,
) -> None:
    """Логирует один run в MLflow: параметры, метрики и (опционально)
    файл-артефакт (например, обученную модель .joblib). Тихо ничего не
    делает, если USE_MLFLOW=False, пакет не установлен или сервер
    недоступен - ни одна из этих ситуаций не должна быть заметна
    вызывающему коду."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=run_name):
            for key, value in (params or {}).items():
                mlflow.log_param(key, value)
            for key, value in (metrics or {}).items():
                if value is not None:
                    mlflow.log_metric(key, value)
            if artifact_path:
                mlflow.log_artifact(artifact_path)
    except Exception as exc:
        logger.warning(f"MLflow logging failed for run '{run_name}': {exc}")
