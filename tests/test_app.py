"""
Smoke tests for the FastAPI app's wiring.

This is the project's first test suite - none existed before, even though
the Makefile and CI both referenced one. Kept deliberately network-free
(no live CBR/Ollama calls) so it runs the same in CI as on a laptop: the
data endpoints are exercised with the loader's built-in fallback behavior,
and the RAG endpoint is exercised with a greeting question, which
short-circuits before any Ollama call.

Every one of these was a real, reproducible bug before this pass, not a
hypothetical one; each test below is a regression test pinned to that bug:

- GET / and GET /monitoring/dashboard used to raise
  `TypeError: unhashable type: 'dict'` on every single request, because
  both called the old, removed `TemplateResponse(name, context)` calling
  convention against a current Starlette that requires
  `TemplateResponse(request, name, context)`.
- GET /api/ab-test/status, GET /api/ping and GET /api/health used to 404:
  their routers were built (src/presentation/ab_testing/routes.py,
  src/presentation/api/routes/health.py) but never mounted in
  src/main.py.
- POST /api/refresh used to raise `TypeError: get_historical_data() got
  an unexpected keyword argument 'refresh'` from inside its background
  task, because DataService.get_historical_data() had no `refresh`
  parameter at all.
- POST /api/rag/ask used to be reachable only at the accidental
  double-prefixed /api/rag/api/ask.
"""
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root_page_renders():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_root_api_message():
    resp = client.get("/api/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pong"


def test_health_page():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_json_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "components" in body


def test_ab_testing_router_is_mounted():
    resp = client.get("/api/ab-test/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "active"


def test_admin_refresh_endpoint():
    resp = client.post("/api/refresh")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_admin_cache_status_endpoint():
    resp = client.get("/api/cache/status")
    assert resp.status_code == 200
    assert "local_cache_size" in resp.json()


def test_monitoring_dashboard_renders():
    resp = client.get("/monitoring/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_monitoring_api_health():
    resp = client.get("/monitoring/api/health")
    assert resp.status_code == 200


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests_total" in resp.content


def test_data_and_forecast_and_stats_endpoints_respond():
    for path in ("/api/data/data", "/api/forecast/forecast?days=1", "/api/stats/stats"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}: {resp.text}"


def test_rag_ask_endpoint_greeting():
    # A greeting short-circuits before any Ollama call, so this stays
    # fast and network-free.
    resp = client.post("/api/rag/ask", json={"question": "Привет!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "greeting"
    assert body["answer"]
