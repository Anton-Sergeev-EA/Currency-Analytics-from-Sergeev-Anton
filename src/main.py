import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from src.core.config import settings
from src.common.logger.logger import setup_logging, logger
from src.presentation.api.routes import health, data, forecast, rag, stats, admin

setup_logging(settings.LOG_LEVEL)
templates = Jinja2Templates(directory="src/presentation/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    for directory in [settings.DATA_DIR, settings.MODELS_DIR, settings.LOG_DIR, settings.CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created: {directory}")

    logger.info("Application started successfully")

    yield

    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced Currency Analysis System with ML and RAG",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(data.router)
app.include_router(forecast.router)
app.include_router(rag.router)
app.include_router(stats.router)
app.include_router(admin.router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def api_root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "operational",
        "docs": "/api/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
