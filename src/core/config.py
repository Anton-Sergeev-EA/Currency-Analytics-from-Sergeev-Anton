"""
Application settings — single source of truth.

Previously this project had two independent `Settings` classes
(`src/config.py` and `src/core/config.py`) with the same class name but
different fields, and only one of them actually read from `.env`. Several
modules referenced settings fields (`APP_VERSION`, `OPENAI_API_KEY`,
`DATA_DIR`, `LLM_MODEL`, `LLM_ENABLED`) that existed in neither class,
which would raise `AttributeError` the moment that code path executed.

This module consolidates everything into one class, reading from `.env`
consistently, so every setting used anywhere in the codebase is declared
exactly once, in exactly one place.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # --- App metadata ---
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8002

    # --- Cache ---
    # There used to be a DATABASE_URL here too (and docker-compose.prod.yml
    # pointed it at a Postgres service), but nothing in the codebase ever
    # opened a connection with it - there is no SQL persistence layer, only
    # this Redis cache and the A/B test module's own separate embedded
    # SQLite file (src/ab_testing/ab_service.py, unrelated to this setting).
    # Removed rather than left in place implying a feature that isn't
    # there; see docker-compose.prod.yml for the matching cleanup.
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Security (change these via .env in production — see .env.example) ---
    SECRET_KEY: str = "change_this_in_production"
    JWT_SECRET_KEY: str = "change_this_in_production"

    # --- External data source ---
    CBR_API_URL: str = "https://www.cbr.ru/scripts/XML_daily.asp"

    # --- Paths ---
    # MODEL_PATH used to live here too, but nothing read it - trainer.py
    # and forecast_service.py both hardcode their own "data/models"
    # default instead. Removed as dead, misleading configuration.
    DATA_PATH: str = "./data"
    DATA_DIR: str = "./data"  # used by the RAG vector store; kept distinct
    # from DATA_PATH in case the two are ever pointed at different
    # locations (e.g. raw currency data vs. the vector DB), but defaults
    # to the same value today.

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Public domain / CORS ---
    # Comma-separated list of origins the browser is allowed to call the
    # API from cross-origin. The templates in src/presentation/templates
    # are served by this same app and call the API with relative paths
    # (same-origin, so browsers never even consult this list for them);
    # this only matters if something else - a separate frontend, a
    # mobile app's webview, a tool hitting the API from another domain -
    # calls it directly. Defaults cover the production domain plus local
    # development; override via .env for a different deployment.
    ALLOWED_ORIGINS: str = "https://anton-analytics.ru,https://www.anton-analytics.ru,http://localhost:8002,http://127.0.0.1:8002"

    # --- Ollama (local LLM) ---
    USE_OLLAMA: bool = True
    OLLAMA_BASE_URL: str = "http://172.17.0.1:11434"
    OLLAMA_MODEL: str = "tinyllama"

    # --- OpenAI-compatible LLM (optional alternative to Ollama) ---
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    LLM_MODEL: str = "tinyllama"
    LLM_ENABLED: bool = True

    # --- MLflow experiment tracking (optional) ---
    # The tracking server itself is NOT part of this repo - it runs as
    # its own systemd service on the VDS (own venv, own sqlite backing
    # store), entirely independent of this app's Docker container. This
    # only points train_models.py at it. 172.17.0.1 is the Docker bridge
    # gateway - the same way this container already reaches Redis and
    # Ollama on the host (see REDIS_URL/OLLAMA_BASE_URL above). Set to ""
    # to disable logging entirely (e.g. local dev with no MLflow server
    # running) - train_models.py treats a blank URI, and any connection
    # failure to a non-blank one, as "skip logging, don't fail training".
    MLFLOW_TRACKING_URI: str = "http://172.17.0.1:5000"
    MLFLOW_EXPERIMENT_NAME: str = "currency-forecast"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
