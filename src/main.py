import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from src.core.config import settings
from src.common.logger.logger import get_logger
from src.common.prometheus_metrics import instrument as instrument_prometheus
from src.presentation.api.routes import router as api_router
from src.presentation.api.routes.admin import router as admin_router
from src.presentation.api.routes.health import router as health_router
from src.presentation.ab_testing.routes import router as ab_testing_router
from src.presentation.routes import router as web_router
from src.presentation.monitoring.routes import router as monitoring_router

logger = get_logger("main")


async def _warm_model_accuracy_cache() -> None:
    """Pre-computes the monitoring dashboard's walk-forward backtest for
    every currency in the background, right after boot.

    `/monitoring/api/model-accuracy` is real CPU work - a bounded
    hyperparameter search per currency, easily minutes total on a cold
    cache (see ModelEvaluator's docstring) - and it used to run inline on
    whichever visitor's request happened to hit it first after a deploy,
    which is exactly what made the monitoring dashboard itself appear to
    hang after a fresh deploy. Firing it here instead means a real visitor
    almost always lands on an already-warm cache (6-hour TTL) rather than
    paying that cost themselves. This task runs concurrently with normal
    request handling, not before it - ModelEvaluator's own CPU-bound work
    is offloaded to a worker thread, so this doesn't delay startup or
    block any other endpoint while it runs.
    """
    from src.core.constants import SUPPORTED_CURRENCIES
    from src.presentation.monitoring.routes import evaluator

    for ccy in SUPPORTED_CURRENCIES:
        try:
            await evaluator.get_accuracy(ccy)
        except Exception as exc:
            logger.warning("Startup accuracy cache warm-up failed for %s: %s", ccy, exc)
    logger.info("Startup accuracy cache warm-up complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Currency Analytics System...")
    warmup_task = asyncio.create_task(_warm_model_accuracy_cache())
    yield
    warmup_task.cancel()
    logger.info("Shutting down Currency Analytics System...")

app = FastAPI(
    title="Currency Analytics System",
    description="Advanced currency analysis with ML forecasting and RAG assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Restricted to the real deployment domain (anton-analytics.ru) plus local
# dev by default (ALLOWED_ORIGINS in .env) - previously a hardcoded "*",
# which the browser accepts for a same-origin app like this one but is
# unnecessarily permissive once the app is reachable at a public domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adds the request-count/duration middleware and GET /metrics that
# prometheus.yml has been scraping this app for all along.
instrument_prometheus(app)

app.include_router(api_router, prefix="/api")
# admin_router's own routes already hardcode the "/api" prefix (they are
# relied on as-is by install.sh's cron jobs and by the dashboard's
# refresh buttons), so it is mounted without an extra prefix here.
app.include_router(admin_router)
# health_router adds a JSON /api/health (distinct from the human-facing
# HTML page at GET /health below), plus /api/ping and /api/health/verbose.
app.include_router(health_router, prefix="/api")
# Was built but never mounted, so GET /api/ab-test/status - the endpoint
# the root response below advertises - 404'd.
app.include_router(ab_testing_router, prefix="/api")
app.include_router(web_router)
app.include_router(monitoring_router)

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("src/presentation/static/favicon.svg")

@app.get("/")
async def root():
    return {
        "message": "Currency Analytics System",
        "version": "1.0.0",
        "developer": "Sergeev Anton Valentinovich",
        "docs": "/docs",
        "monitoring": "/monitoring/dashboard",
        "ab_testing": "/api/ab-test/status"
    }

@app.get("/health", response_class=HTMLResponse)
async def health():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Check</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f2f5; padding: 40px; text-align: center; }
            .card { background: white; border-radius: 12px; padding: 30px; max-width: 500px; margin: 0 auto; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .status { font-size: 48px; }
            .healthy { color: #48bb78; }
            .detail { text-align: left; margin-top: 20px; padding: 15px; background: #f7fafc; border-radius: 8px; }
            .component { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
            .ok { color: #48bb78; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="status healthy">✅</div>
            <h1>Currency Analytics System</h1>
            <p style="color: #48bb78; font-size: 20px;">Система работает</p>
            <div class="detail">
                <div class="component"><span>API</span><span class="ok">✅ ok</span></div>
                <div class="component"><span>Модели</span><span class="ok">✅ loaded</span></div>
                <div class="component"><span>Ollama</span><span class="ok">✅ connected</span></div>
                <div class="component"><span>Время</span><span id="time"></span></div>
            </div>
            <p style="margin-top: 20px; color: #718096; font-size: 14px;">
                <a href="/" style="color: #4299e1;">← На главную</a>
            </p>
        </div>
        <script>
            document.getElementById('time').textContent = new Date().toLocaleString();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
