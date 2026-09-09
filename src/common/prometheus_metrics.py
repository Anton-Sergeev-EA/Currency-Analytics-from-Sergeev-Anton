"""
Prometheus instrumentation.

`prometheus.yml` (repo root) has always scraped this app at
`currency-analytics:8000/metrics`, but no `/metrics` route - or any
request instrumentation - ever existed. There *was* a `src/monitoring/
metrics.py` that defined the metrics and a middleware, but it referenced
an undefined module-level `app`/`Request` (it was written to be pasted
directly into main.py, never actually wired up) and `prometheus_client`
wasn't even a project dependency, so importing that module raised
`NameError` immediately. It was dead, broken code and has been removed;
this module replaces it with something that is actually mounted and
actually works.
"""
import time

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)


def instrument(app: FastAPI) -> None:
    """Add request-metrics middleware and a GET /metrics route to `app`."""

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start_time = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start_time

        # request.url.path would create a new label series per dynamic
        # path segment (e.g. every distinct currency/date) if used as-is;
        # the matched route template is stable and keeps cardinality low.
        route = request.scope.get("route")
        endpoint = route.path if route is not None else request.url.path

        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()
        REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
