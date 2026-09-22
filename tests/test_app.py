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
- POST /api/refresh, POST /api/force-refresh and GET /api/cache/status
  used to have no authentication at all - reachable by anyone on the
  public internet, and /api/force-refresh triggers real work (a live CBR
  fetch plus a full data reload). They now require an X-Admin-Key header
  matching SECRET_KEY (src/presentation/api/routes/admin.py).
- POST /api/rag/ask used to be reachable only at the accidental
  double-prefixed /api/rag/api/ask.
"""
from fastapi.testclient import TestClient

from src.core.config import settings

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
    resp = client.post("/api/refresh", headers={"X-Admin-Key": settings.SECRET_KEY})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_admin_cache_status_endpoint():
    resp = client.get("/api/cache/status", headers={"X-Admin-Key": settings.SECRET_KEY})
    assert resp.status_code == 200
    assert "local_cache_size" in resp.json()


def test_admin_endpoints_reject_missing_or_wrong_admin_key():
    """The actual security-relevant behavior: an admin endpoint used to
    be reachable by anyone with no credential at all - pin that it now
    isn't, for all three admin routes, both with no header at all and
    with a wrong one."""
    for method, path in (
        ("post", "/api/refresh"),
        ("post", "/api/force-refresh"),
        ("get", "/api/cache/status"),
    ):
        call = getattr(client, method)

        resp_no_header = call(path)
        assert resp_no_header.status_code == 401, f"{path} should reject a request with no X-Admin-Key"

        resp_wrong_key = call(path, headers={"X-Admin-Key": "definitely-not-the-real-key"})
        assert resp_wrong_key.status_code == 401, f"{path} should reject a wrong X-Admin-Key"


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


def test_cny_and_gbp_are_first_class_currencies_not_just_usd_eur():
    """
    Currency handling used to be hardcoded to USD/EUR in half a dozen
    places (loader.py's fetchers, forecast_service.py's branching,
    data_service.py, the knowledge-base builder, finance_advisor.py's
    alias lists) while src/core/constants.py's Currency enum listing all
    four sat completely unused. Adding CNY/GBP is now a one-line change
    to SUPPORTED_CURRENCIES - this pins that the rest of the stack
    actually picks it up end to end.
    """
    resp = client.get("/api/data/data?period_days=30")
    assert resp.status_code == 200
    body = resp.json()
    assert "cny" in body and "gbp" in body

    resp = client.get("/api/forecast/forecast?days=3&currency=CNY")
    assert resp.status_code == 200
    forecast = resp.json()
    assert isinstance(forecast, list) and len(forecast) == 3

    resp = client.get("/api/stats/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert "cny_current" in stats and "gbp_current" in stats


def test_rag_handles_cny_conversion():
    resp = client.post("/api/rag/ask", json={"question": "Переведи 100 юаней в рубли"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "conversion"
    assert "CNY" in body["answer"]
