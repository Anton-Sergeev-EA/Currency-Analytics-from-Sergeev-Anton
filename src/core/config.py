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

    # --- Database / cache ---
    DATABASE_URL: str = "sqlite:///./data/analytics.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Security (change these via .env in production — see .env.example) ---
    SECRET_KEY: str = "change_this_in_production"
    JWT_SECRET_KEY: str = "change_this_in_production"

    # --- External data source ---
    CBR_API_URL: str = "https://www.cbr.ru/scripts/XML_daily.asp"

    # --- Paths ---
    MODEL_PATH: str = "./models"
    DATA_PATH: str = "./data"
    DATA_DIR: str = "./data"  # used by the RAG vector store; kept distinct
    # from DATA_PATH in case the two are ever pointed at different
    # locations (e.g. raw currency data vs. the vector DB), but defaults
    # to the same value today.

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Ollama (local LLM) ---
    USE_OLLAMA: bool = True
    OLLAMA_BASE_URL: str = "http://172.17.0.1:11434"
    OLLAMA_MODEL: str = "tinyllama"

    # --- OpenAI-compatible LLM (optional alternative to Ollama) ---
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    LLM_MODEL: str = "tinyllama"
    LLM_ENABLED: bool = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
