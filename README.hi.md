# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · [中文](README.zh.md) · **हिन्दी** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

Russian central bank डेटा से USD/RUB, EUR/RUB, CNY/RUB और GBP/RUB विश्लेषण का research application। अतिरिक्त पोर्टफोलियो परियोजना; अभी सुविधाओं का विस्तार नहीं।

## सुविधाएँ और तकनीक

Python 3.10+, FastAPI, pandas, NumPy, LightGBM, XGBoost और scikit-learn; HTML/CSS/JS तथा local Chart.js। Trained model न होने पर statistical trend fallback है। TF-IDF knowledge retrieval, code से numerical calculations और Ollama से खुले प्रश्नों के उत्तर; sources में retrieved documents हैं। Redis अनुपलब्ध होने पर process-memory cache है। SQLite/SQLAlchemy में A/B assignment और forecast comparison logs हैं। Prometheus, JSON logs और optional Grafana monitoring उपलब्ध हैं।

## मूल्यांकन की सीमाएँ

Unit/smoke tests persistence baseline (कल का rate आज जैसा), trading profit या production deployment पर superiority सिद्ध नहीं करते। Code में walk-forward evaluation है; विश्वसनीय तुलना के लिए chronological held-out data, hashes, configuration और raw results चाहिए। A/B assignment, MAE/MAPE या t-test अकेले valid experiment नहीं; time dependence और repeated observations को ध्यान में रखना चाहिए। Intervals स्वतंत्र reliability evidence नहीं और निवेश निर्णय का आधार नहीं होने चाहिए।

## स्थापना, संचालन और परीक्षण

पहले deployment के लिए environment variables और अपने SECRET_KEY/JWT_SECRET_KEY सेट करें। Repository में configuration template है; secrets Git में न डालें। Commands उदाहरण हैं, deployment evidence नहीं। CI Python 3.10/3.11 जाँचता है; tests live central bank या Ollama service पर निर्भर नहीं हैं।

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m uvicorn src.main:app --host 127.0.0.1 --port 8002
```

```sh
docker compose up -d --build
curl http://localhost:8002/api/health
docker compose logs -f
```

```sh
.venv/bin/python train_models.py
.venv/bin/python scripts/update_ab_actual_rates.py
```

## API और उदाहरण

Historical data, forecast, stats, assistant और health paths नीचे हैं। /api/refresh, /api/force-refresh और /api/cache/status को X-Admin-Key चाहिए। Assistant response में sources हैं। /api/ab-test/* assignment/logging देता है; refresh-actuals से actual rates अपडेट किए बिना तुलना सार्थक नहीं।

```text
GET  /api/data/data?period_days=30
GET  /api/forecast/forecast?days=7&currency=USD
GET  /api/stats/stats
GET  /api/health
GET  /api/ping
POST /api/rag/ask
GET  /api/ab-test/status
POST /api/ab-test/predict?currency=usd_rate&days=1
GET  /api/ab-test/stats?days=30
POST /api/ab-test/update-ratios?split_a=0.5
POST /api/ab-test/refresh-actuals
```

```sh
curl 'http://localhost:8002/api/forecast/forecast?days=7&currency=USD'
curl -X POST http://localhost:8002/api/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the USD forecast for next week?"}'
```

## संचालन और सीमाएँ

Compose host port 8002 प्रकाशित करता है, default केवल loopback नहीं है; deployment से पहले network access सीमित करें। /monitoring/dashboard monitoring UI, /metrics Prometheus और /docs तथा /redoc API docs देते हैं। Ollama और MLflow अलग services हैं; खाली MLFLOW_TRACKING_URI tracking बंद करता है। 768 MiB limit, single worker और 4 GiB VDS लक्ष्य resource benchmarks नहीं। monitoring profile optional है। Compose root-user override, security config, dependencies, backup और HTTPS deployment पर जाँचना आवश्यक है।

## भाषाएँ और लाइसेंस

ये संक्षिप्त localized guides हैं; विस्तृत उदाहरण English या Russian README में हैं। Application UI, assistant और logs को नई language support नहीं मिली। License की मूल शर्तें LICENSE में हैं; forecasts केवल information और education के लिए हैं।

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
