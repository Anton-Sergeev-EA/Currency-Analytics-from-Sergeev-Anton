import os
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.presentation.api.routes import data, forecast, stats, health, rag, admin

app = FastAPI(
    title="Currency Analytics Pro",
    version="2.0.0",
    description="Advanced Currency Analysis System with ML and RAG",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs"
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "presentation" / "templates"
STATIC_DIR = BASE_DIR / "presentation" / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(data.router)
app.include_router(forecast.router)
app.include_router(stats.router)

if hasattr(health, 'router'):
    app.include_router(health.router)
if hasattr(rag, 'router'):
    app.include_router(rag.router)
if hasattr(admin, 'router'):
    app.include_router(admin.router)

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
