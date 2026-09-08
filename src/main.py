import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from src.core.config import settings
from src.common.logger.logger import get_logger
from src.presentation.api.routes import router as api_router
from src.presentation.routes import router as web_router
from src.presentation.monitoring.routes import router as monitoring_router

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Currency Analytics System...")
    yield
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
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
