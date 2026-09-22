# Currency Analytics

Русская версия: [README.ru.md](README.ru.md)

A web application for analyzing and forecasting USD/RUB, EUR/RUB, CNY/RUB and GBP/RUB
exchange rates using official Bank of Russia data. Includes an ensemble
ML forecasting model (LightGBM, XGBoost, Random Forest, Gradient
Boosting), a RAG assistant on a local Ollama model, an interactive
dashboard, and a real A/B test of the model against a statistical
baseline.

The project is deliberately sized to run on a **small VDS (4GB RAM,
limited disk)** — see [Deploying on a small VDS](#deploying-on-a-small-vds).

## Features

- Historical exchange rates from the Bank of Russia API (USD/RUB,
  EUR/RUB, CNY/RUB, GBP/RUB), with a fallback source and a statistical trend if the
  primary source is unavailable.
- Ensemble ML forecasting with confidence intervals; falls back to a
  robust statistical trend when no trained model exists yet for a
  currency.
- **RAG assistant with real retrieval**: TF-IDF search over a knowledge
  base (deliberately no torch/sentence-transformers/chromadb — a much
  lighter, equally legitimate method for a corpus this small), plus
  **deterministic financial calculations** (currency conversion,
  investment projections, currency comparisons) — numbers are computed
  by code, never guessed by the small local LLM. Ollama is only used for
  what it's actually good at: phrasing an answer to open-ended questions
  from retrieved context. The API response includes `sources` — which
  knowledge-base documents were actually used.
- Interactive web dashboard (Chart.js) with a currency converter and a
  dark theme.
- Redis-backed caching with automatic fallback to an in-memory cache if
  Redis is unavailable.
- **Real A/B testing** (`/api/ab-test/*`): variant A is the real ML
  ensemble, variant B is a persistence/random-walk forecast (the
  standard benchmark in FX forecasting). Deterministic user-to-variant
  assignment, a SQLite log, real MAE/MAPE and a significance t-test once
  the actual rate becomes known (`POST /api/ab-test/refresh-actuals`).
- **Real monitoring dashboard metrics**: model accuracy is a genuine
  walk-forward backtest (train excluding the last N days, evaluate only
  on those), not random numbers; data quality and the live prediction
  panel are computed from actual data.
- Prometheus metrics (`/metrics`) and an optional Grafana/Prometheus
  Compose profile.
- Structured JSON logging.
- A pytest smoke-test suite and GitHub Actions CI.

## Model accuracy — read this before trusting a forecast

Day-ahead FX rates are close to a random walk: a huge share of a currency's
next-day move is, genuinely, not predictable from its own history. Because
of that, the honest bar for "the model is useful" is not a high R² in
isolation - it's **beating a naive persistence baseline** ("tomorrow's rate
= today's rate"), which is already a strong, hard-to-beat benchmark for
this kind of series. Any dashboard, paper, or product that reports FX
forecast accuracy without that comparison is not telling you anything
verifiable.

This project reports that comparison directly, computed the same way in
three places: `train_models.py`'s console output after every training run,
`/monitoring/api/model-accuracy`, and `GET /api/health`'s `components`. All
three use the same walk-forward backtest (`ModelEvaluator`): train on data
strictly *before* a held-out window, predict that window, never let the
model see it during training - and average the result over several
non-overlapping held-out windows rather than just the most recent one, so
one lucky or unlucky window doesn't flip the headline verdict by chance.

**Current honest state, on real 3-years-of-history Bank of Russia data:
none of the four currencies beat the naive baseline outright.** As of the
last real training run, GBP comes closest (backtest RMSE 0.865 vs. the
naive baseline's 0.855 - within a hair of it); USD, EUR and CNY trail by a
larger margin. This is reported plainly by the same backtest referenced
above, not hidden or rounded away - and it is a realistic outcome for a
model trained on public data with no proprietary order-flow or
high-frequency signal, not a bug to be silently patched away with a
better-looking number. If you retrain this yourself, check your own
console output rather than trusting the numbers above - they will already
be out of date the moment the model is retrained on a different window of
history.

Two engineering decisions came directly out of chasing this gap honestly,
instead of covering it up:

- **Two candidate features were tried and reverted.** A rate-change-recency
  feature and a return-volatility feature both looked reasonable, but a
  real backtest on live data showed they increased error on 3 of 4
  currencies - reverted, with the reasoning and the before/after numbers
  kept as a comment in `engineer.py` and pinned by a test in
  `tests/test_ml.py`, instead of quietly dropped.
- **The ensemble now includes the naive persistence forecast itself** as a
  fifth, zero-parameter candidate, weighted by the same held-out
  cross-validation rule as the four tree-based models (see
  `EnsembleModel`'s docstring) - a standard technique in forecasting
  (blending a naive/statistical baseline with ML, the way winning entries
  in competitions like the M4 do, rather than treating them as rivals).
  This measurably narrowed the gap to the naive baseline on all four
  currencies compared to ML-only, without fully closing it: the naive
  candidate typically earns 35-60% of the ensemble's weight, but the rest
  still goes to the four ML models, which are individually behind the
  baseline on this data - so the blend improves on ML-only without being
  mathematically guaranteed to beat pure persistence outright. Not a trick
  to disguise a weak model as a strong one: the weight is earned purely by
  measured performance, applied identically in training, in the saved
  production model, and in this backtest.

## Tech stack

**Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy
(A/B test log), Redis, aiohttp.
**ML:** LightGBM, XGBoost, scikit-learn, pandas, NumPy.
**AI/RAG:** Ollama (local LLM), a scikit-learn TF-IDF retriever
(deliberately not torch/sentence-transformers/chromadb — see below).
**Frontend:** plain HTML/CSS/JS + Chart.js, no framework.
**Infra:** Docker (multi-stage build), Docker Compose, Prometheus,
Grafana (optional), nginx (prod).

## Deploying on a small VDS

The project defaults are tuned for a **4GB-RAM VDS with limited disk**:

- No heavy embedding stack (torch/sentence-transformers/chromadb) —
  a lightweight TF-IDF retriever (scikit-learn, already needed for
  forecasting) is used instead. Saves roughly 1.5-2GB of disk and
  hundreds of MB of RAM.
- The Docker image is built in two stages (see `Dockerfile`): compilers
  (gcc/g++/make) stay in the throwaway build stage and never reach the
  final image.
- **1 uvicorn worker by default** (`UVICORN_WORKERS=1` in
  `docker-compose.yml`) — every extra worker is a whole extra copy of
  pandas/numpy/lightgbm/xgboost in RAM.
- A hard 768MB memory cap on the app container (`mem_limit` in
  `docker-compose.yml`).
- `docker-compose.prod.yml` defaults to **1 replica** (`REPLICAS=1`)
  instead of 2 — only raise it once `free -h` shows real headroom.
- Use `tinyllama` or another 1-2B model for the assistant — bigger
  models compete with the app itself for the same RAM.
- The Prometheus/Grafana stack is entirely optional
  (`--profile monitoring`) and not required to run the app — the
  built-in `/monitoring/dashboard` is usually enough on a small VDS.
- Consider setting up **swap** (2GB+) on the VDS if you haven't, for
  short-lived spikes (e.g. Ollama inference while a model is training).

## Deploying on a domain (anton-analytics.ru) with HTTPS

The project is configured to be reachable at **https://anton-analytics.ru**.
This describes the common case: a **shared VDS** that already runs a
host-level nginx and already holds a valid Let's Encrypt certificate for
the domain (e.g. because another project or a previous version of this
app lives on the same server) - not an HTTPS bootstrap from scratch.

1. **Clone the repo and start the stack on its own port:**

   ```bash
   git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
   cd Currency-Analytics-from-Sergeev-Anton
   cp .env.example .env   # review SECRET_KEY/JWT_SECRET_KEY before real use
   docker compose up -d --build
   ```

   The app listens on `127.0.0.1:8002` (see `docker-compose.yml`), so it
   never competes for ports 80/443 with anything else already running
   on the server.

2. **Point the existing nginx site at it.** Edit the site's config
   (e.g. `/etc/nginx/sites-available/anton-analytics`) so its
   `proxy_pass` targets `http://127.0.0.1:8002`, then:

   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```

   No certbot run is needed here - the certificate already covers the
   domain.

3. **Verify, then retire the previous version:**

   ```bash
   curl -I https://anton-analytics.ru/health
   ```

   Open https://anton-analytics.ru and
   https://anton-analytics.ru/monitoring/dashboard, confirm both work,
   and only then stop whatever service used to answer on this domain.

### Startup and recovery on reboot

- The `currency-analytics` container runs with `restart: unless-stopped`
  (`docker-compose.yml`) and a `healthcheck` (`GET /health` every 30s) -
  Docker restarts it on crash and reports it unhealthy on failure.
- `systemctl enable docker` (usually already on for a server that's
  been running other Docker workloads) makes sure the daemon - and
  every `restart: unless-stopped` container - comes back after a
  reboot.
- CORS is controlled by `ALLOWED_ORIGINS` in `.env`/`docker-compose.yml`
  (defaults to the production domain + local dev), not a hardcoded `*`.

Useful commands:

```bash
docker compose ps
docker compose logs -f
sudo nginx -T | grep -A5 anton-analytics   # confirm which port nginx proxies to
```

> Provisioning HTTPS on a bare server with no existing nginx or
> certificate is a different, less common scenario. `docker-compose.prod.yml`
> and `scripts/init_letsencrypt.sh` are kept in the repo for that case,
> but are not part of the anton-analytics.ru deployment described above.

## Quick start (Docker)

```bash
git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
cd Currency-Analytics-from-Sergeev-Anton
cp .env.example .env

docker-compose up -d

curl http://localhost:8002/health
```

Then open:
- Main UI: http://localhost:8002
- Monitoring dashboard: http://localhost:8002/monitoring/dashboard
- API docs (Swagger UI): http://localhost:8002/docs

For the assistant, install and run [Ollama](https://ollama.com) separately:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama
ollama serve
```

`tinyllama` is the default (`OLLAMA_MODEL` in `.env`) and the
recommended choice for a small VDS; larger models answer more
accurately but need more RAM and are slower.

## Local install (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env

uvicorn src.main:app --host 0.0.0.0 --port 8002
```

To have forecasts use the real ML model instead of the trend fallback,
train the models once on historical data:

```bash
python train_models.py
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run fully offline (no live calls to the CBR API or Ollama) — this
is what CI runs on every push.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard. |
| GET | `/health` | Human-readable health check page. |
| GET | `/api/health` | Machine-readable health check (JSON). |
| GET | `/api/ping` | Liveness check. |
| GET | `/api/data/data?period_days=N` | Historical exchange rates. |
| GET | `/api/forecast/forecast?days=N&currency=USD\|EUR\|CNY\|GBP\|ALL` | 1-30 day ML forecast. |
| POST | `/api/rag/ask` | Ask the assistant a question (the response includes `sources` — knowledge-base documents actually used). |
| GET | `/api/stats/stats` | Current rates and day-over-day change. |
| POST | `/api/refresh` | Background data refresh. **Requires header** `X-Admin-Key: <SECRET_KEY>`. |
| POST | `/api/force-refresh` | Synchronous forced data refresh, bypassing the cache. **Requires header** `X-Admin-Key: <SECRET_KEY>`. |
| GET | `/api/cache/status` | Cache status. **Requires header** `X-Admin-Key: <SECRET_KEY>`. |
| GET | `/api/ab-test/status` | A/B test status and current traffic split. |
| POST | `/api/ab-test/predict?currency=usd_rate&days=1` | Real forecast through the assigned variant (A/B), logged to the DB. |
| GET | `/api/ab-test/stats?days=30` | Real per-variant MAE/MAPE + significance t-test. |
| POST | `/api/ab-test/update-ratios?split_a=0.5` | Change variant A's traffic share. |
| POST | `/api/ab-test/refresh-actuals` | Fetch actual rates for forecasts whose date has passed (needed before stats are meaningful). |
| GET | `/monitoring/dashboard` | Monitoring dashboard UI. |
| GET | `/monitoring/api/health` \| `/models` \| `/current-rates` \| `/model-accuracy` \| `/prediction-test` \| `/data-quality` | Dashboard data — all computed from real data/models (see `ModelEvaluator`), with a multi-hour cache on the expensive ones. |
| GET | `/metrics` | Prometheus metrics. |
| GET | `/docs`, `/redoc` | OpenAPI docs. |

### Examples

```bash
# 7-day USD forecast
curl "http://localhost:8002/api/forecast/forecast?days=7&currency=USD"

# Ask the assistant
curl -X POST http://localhost:8002/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the USD forecast for next week?"}'

# A real calculation: projected profit
curl -X POST http://localhost:8002/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько я заработаю, если вложу 100000 рублей в евро на месяц?"}'

# Historical data
curl "http://localhost:8002/api/data/data?period_days=30"
```

## Operations

```bash
make up            # docker-compose up -d
make deploy        # production config (docker-compose.prod.yml)
make logs-app      # application logs
make backup        # archive the data/ directory (including trained models)
make test          # pytest tests/
```

To keep A/B test accuracy stats current, schedule (see `cron_setup.txt`):

```bash
python scripts/update_ab_actual_rates.py
```

`docker-compose.prod.yml` adds nginx in front of the app;
`prometheus`/`grafana` are an optional Compose profile (`monitoring`),
not required for the app to work.

## Security notes

A real security review of this codebase found and fixed one concrete
issue: `/api/refresh`, `/api/force-refresh` and `/api/cache/status` used
to be reachable by anyone on the public internet with no credential at
all, and `/api/force-refresh` triggers genuine work (a live Bank of
Russia fetch plus a full data reload) with no rate limiting - trivially
abusable by anyone who read this README. They now require a
`X-Admin-Key` header matching `SECRET_KEY` from `.env` (see
`require_admin_key` in `src/presentation/api/routes/admin.py`). If
`SECRET_KEY` is still the default placeholder from `.env.example`, the
app logs a loud warning on startup - that value is public in this repo,
so the check runs but protects nothing until you set a real one.

A second real issue was found and fixed the same way: `pip-audit -r
requirements.txt` flagged known CVEs in `aiohttp` and, transitively, in
`starlette` - `fastapi` was pinned to a version whose own dependency
range capped `starlette` below the fix. Fixed by pinning `fastapi` to
0.141.1 (which lifts that cap), pinning `starlette` directly to 1.3.1 -
the earliest release with every flagged CVE resolved - and bumping
`aiohttp`, `lxml`, `jinja2`, `python-dotenv` and `lightgbm` to their
current stable releases. The full test suite (33 tests) passes against
these versions, and `pip-audit -r requirements.txt` currently reports
zero known vulnerabilities. This will go stale again - re-run
`pip-audit` occasionally and check its output against the test suite
before bumping anything further.

One thing found but deliberately **not** silently changed, since fixing
it wrong could be worse than leaving it documented:

- **The container currently runs as root** (`user: "0:0"` in
  `docker-compose.yml`), even though the `Dockerfile` creates and
  switches to an unprivileged `analytics` user - the compose override
  is almost certainly there because the bind-mounted `./data` and
  `./logs` are owned by root on the host. Fixing this properly means
  changing ownership of those host directories (or matching the
  container user's UID/GID to them) - a coordinated change on the real
  server, not something to flip in one file without checking what
  actually owns those paths first. Until then, a compromise of the app
  process has more privilege inside the container than it needs to.

Not found: no raw SQL anywhere in this codebase (the only database
access, the A/B test log, goes entirely through SQLAlchemy's ORM query
builder), and no `eval`/`exec`/`pickle.loads`/`os.system` calls.

## License

MIT with mandatory attribution — see [LICENSE](LICENSE).

## Disclaimer

Provided for informational and educational purposes. Forecasts are not
financial advice; consult a professional before making investment
decisions based on this application's output.

## Acknowledgements

- The Bank of Russia, for exchange rate data.
- The open-source projects this application is built on.

---

Anton Sergeev — avsergeev1981@gmail.com
