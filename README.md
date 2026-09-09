# Currency Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)

Русская версия: [README.ru.md](README.ru.md)

A web application for analyzing and forecasting USD/RUB and EUR/RUB exchange
rates, built on official Bank of Russia data. It combines an ensemble ML
forecaster (LightGBM, XGBoost, Random Forest, Gradient Boosting), a
RAG-based chat assistant backed by a local Ollama model, an interactive
dashboard, and a small A/B-testing API.

## Features

- Historical exchange rate data from the CBR API (USD/RUB, EUR/RUB), with a
  scraped-secondary-source and statistical-trend fallback when the primary
  source is unreachable.
- Ensemble ML forecasting with confidence intervals, falling back to a
  trend-based estimate when no trained model is available for a currency.
- A RAG-based chat assistant (Ollama, local) for questions about rates and
  forecasts.
- An interactive web dashboard with live charts (Chart.js).
- Redis-backed caching with an automatic in-memory fallback when Redis is
  unreachable.
- A demo A/B-testing API (`/api/ab-test/*`) — currently fixed mock
  responses, not wired to live traffic splitting.
- Prometheus metrics (`/metrics`) and a Grafana/Prometheus Compose profile.
- Structured JSON logging.
- A pytest smoke-test suite and GitHub Actions CI.

## Tech stack

**Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy
(A/B-test log storage only), Redis, aiohttp.
**ML:** LightGBM, XGBoost, scikit-learn, pandas, NumPy.
**AI/RAG:** Ollama (local LLM), Sentence-Transformers, ChromaDB.
**Frontend:** vanilla HTML/CSS/JS + Chart.js — no frontend framework.
**Infra:** Docker, Docker Compose, Prometheus, Grafana, nginx (prod).

## Quick start (Docker)

```bash
git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
cd Currency-Analytics-from-Sergeev-Anton
cp .env.example .env

docker-compose up -d

curl http://localhost:8002/health
```

Then open:
- Main interface: http://localhost:8002
- Monitoring dashboard: http://localhost:8002/monitoring/dashboard
- API docs (Swagger UI): http://localhost:8002/docs

For the chat assistant, install and run [Ollama](https://ollama.com) separately:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama
ollama serve
```

`tinyllama` is the app's default model (`OLLAMA_MODEL` in `.env`); any
Ollama-served model works, larger ones just answer more slowly.

## Local install (no Docker)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env

uvicorn src.main:app --host 0.0.0.0 --port 8002
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The suite runs fully offline (no live CBR or Ollama calls needed) and is
what CI runs on every push. See `tests/test_app.py`'s module docstring for
what each test guards against — several are regression tests for bugs that
made specific endpoints unreachable before this pass.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard. |
| GET | `/health` | Human-readable health page. |
| GET | `/api/health` | Machine-readable health check (JSON). |
| GET | `/api/ping` | Liveness check. |
| GET | `/api/data/data?period_days=N` | Historical exchange rates. |
| GET | `/api/forecast/forecast?days=N&currency=USD\|EUR\|ALL` | ML forecast, 1-30 days. |
| POST | `/api/rag/ask` | Ask the chat assistant a question. |
| GET | `/api/stats/stats` | Current rates and day-over-day change. |
| POST | `/api/refresh` | Trigger a background data refresh (used by the dashboard's "Refresh" button and by `install.sh`'s cron jobs). |
| POST | `/api/force-refresh` | Force a synchronous data refresh, bypassing cache. |
| GET | `/api/cache/status` | Cache backend status. |
| GET | `/api/ab-test/status` \| `/predict` \| `/stats` \| POST `/update-ratios` | A/B-testing demo endpoints (mock data). |
| GET | `/monitoring/dashboard` | Monitoring dashboard UI. |
| GET | `/monitoring/api/health` \| `/models` \| `/current-rates` \| `/model-accuracy` \| `/prediction-test` \| `/data-quality` | Dashboard data feeds. Several of these (`model-accuracy`, `prediction-test`) return randomized illustrative numbers rather than a live model evaluation — they're marked as such in the code. |
| GET | `/metrics` | Prometheus metrics. |
| GET | `/docs`, `/redoc` | OpenAPI docs. |

### Examples

```bash
# 7-day USD forecast
curl "http://localhost:8002/api/forecast/forecast?days=7&currency=USD"

# Ask the assistant
curl -X POST http://localhost:8002/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какой прогноз по доллару на следующую неделю?"}'

# Historical data
curl "http://localhost:8002/api/data/data?period_days=30"
```

### Example questions for the assistant

- "Какой прогноз по доллару на следующую неделю?"
- "Какой прогноз по евро?"
- "Сколько я заработаю, если вложу 100 000 рублей в евро?"
- "Сравни доллар и евро"
- "Какая валюта лучше для инвестиций?"

## Project structure

```
Currency-Analytics-from-Sergeev-Anton/
├── src/
│   ├── ab_testing/          # Standalone A/B-test service + SQLite log store
│   │                        # (used by the ops scripts, not by the mounted
│   │                        # /api/ab-test/* router - see routes' docstrings).
│   ├── application/
│   │   └── services/        # Business-logic services (data, forecast, RAG).
│   ├── common/
│   │   ├── logger/          # Structured JSON logging.
│   │   └── prometheus_metrics.py
│   ├── core/                # Settings (single source of truth), constants.
│   ├── domain/               # Entities and value objects.
│   ├── infrastructure/
│   │   ├── data/             # CBR data loading + Redis/in-memory cache.
│   │   ├── ml/                # Feature engineering, ensemble model, training.
│   │   └── rag/                # Ollama client, vector store, knowledge base.
│   ├── presentation/
│   │   ├── api/routes/        # /api/* routers.
│   │   ├── ab_testing/        # /api/ab-test/* router (mock data).
│   │   ├── monitoring/        # /monitoring/* router + dashboard template.
│   │   └── templates/         # Web dashboard (index.html).
│   └── main.py                # FastAPI app + router wiring.
├── scripts/                   # Cron/ops scripts (data refresh, A/B stats).
├── tests/                     # pytest suite.
├── train_models.py            # CLI: train the ensemble models.
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml    # Adds nginx + horizontal scaling.
├── requirements.txt
├── requirements-dev.txt       # + pytest, pyflakes, and script-only deps.
├── .env.example
├── LICENSE
├── README.md
└── README.ru.md
```

## Deployment / operations

```bash
make up            # docker-compose up -d
make deploy        # production overlay (docker-compose.prod.yml), scaled
make logs-app      # tail the app's logs
make backup        # tar the data/ directory (includes trained models)
make test          # pytest tests/
```

`docker-compose.prod.yml` adds nginx and scales the app to 2 replicas
behind it; `prometheus`/`grafana` are an optional `monitoring` Compose
profile (`docker-compose --profile monitoring up -d`).

## License

MIT, with an attribution requirement — see [LICENSE](LICENSE).

## Disclaimer

Provided for informational and educational purposes. Forecasts are not
financial advice; consult a qualified professional before making
investment decisions based on this software's output.

## Acknowledgements

- The Bank of Russia, for the exchange rate data.
- The open-source projects this is built on.

---

Anton Sergeev — avsergeev1981@gmail.com
