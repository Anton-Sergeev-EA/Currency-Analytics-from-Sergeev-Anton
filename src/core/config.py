import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "Currency Analytics Pro"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CBR_API_URL: str = "http://www.cbr.ru/scripts/XML_daily.asp"

    DEFAULT_PERIOD_DAYS: int = 365
    FORECAST_DAYS: int = 30
    FEATURE_LAGS: List[int] = [1, 3, 7, 14, 30]
    ROLLING_WINDOWS: List[int] = [7, 14, 30]

    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = 60
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    DATA_DIR: str = "data"
    MODELS_DIR: str = "data/models"
    CACHE_DIR: str = "data/cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
