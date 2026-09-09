from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter(tags=["web"])

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@router.get("/", response_class=HTMLResponse)
async def web_index(request: Request):
    # Jinja2Templates.TemplateResponse() takes `request` as its first
    # positional argument in current Starlette; the old two-argument form
    # (name, {"request": request}) silently swaps `name` and `context`
    # into the wrong parameters and raises "unhashable type: dict" from
    # inside Jinja2's template cache the moment this route is hit.
    return templates.TemplateResponse(request, "index.html", {"request": request})
