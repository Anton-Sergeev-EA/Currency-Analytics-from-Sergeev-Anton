from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
MODEL_PREDICTION_COUNT = Counter('model_predictions_total', 'Total model predictions', ['model'])
MODEL_PREDICTION_DURATION = Histogram('model_prediction_duration_seconds', 'Model prediction duration', ['model'])
CBR_API_CALLS = Counter('cbr_api_calls_total', 'Total CBR API calls', ['status'])
DATA_QUALITY_SCORE = Gauge('data_quality_score', 'Data quality score')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
