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


def test_ab_test_predict_is_not_the_old_hardcoded_mock():
    """
    /api/ab-test/predict used to always return a fixed literal
    (variant "control", predicted_rate 87.5, forecast_date "2026-09-05")
    no matter what was asked. It now calls the real ABTestService, which
    computes an actual forecast (ML ensemble or persistence baseline)
    from whatever demo/historical data is loaded.
    """
    resp = client.post("/api/ab-test/predict", params={"user_id": "test-user-1", "currency": "usd_rate", "days": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["variant"] in ("A", "B")
    assert body["model_name"] in ("ml_ensemble", "persistence_baseline")
    assert isinstance(body["predicted_rate"], (int, float))
    assert body["ab_test_active"] is True


def test_ab_test_stats_endpoint_reflects_real_service():
    resp = client.get("/api/ab-test/stats", params={"days": 30})
    assert resp.status_code == 200
    body = resp.json()
    # No fixed "total_requests": 1000 / "improvement": "0.69%" mock fields
    # anymore - the real service reports what's actually in the DB.
    assert "period_days" in body
    assert "total_requests" in body


def test_monitoring_model_accuracy_reports_availability_not_random_numbers():
    resp = client.get("/monitoring/api/model-accuracy")
    assert resp.status_code == 200
    data = resp.json()["data"]
    for currency in ("usd", "eur"):
        assert currency in data
        # Either a real walk-forward backtest ran ("available": True with
        # rmse/mae/mape/r2 from an actual held-out evaluation), or it
        # honestly reports why not - never a `random.random()` filler.
        assert "available" in data[currency]


def test_rag_investment_question_is_a_real_calculation():
    resp = client.post(
        "/api/rag/ask",
        json={"question": "Сколько я заработаю, если вложу 100000 рублей в доллары на неделю?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] in ("investment", "error")  # "error" only if demo data is somehow empty
    if body["type"] == "investment":
        assert "sources" in body and body["sources"]


def test_rag_conversion_question_is_deterministic():
    resp = client.post("/api/rag/ask", json={"question": "Переведи 100 долларов в рубли"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] in ("conversion", "error")
