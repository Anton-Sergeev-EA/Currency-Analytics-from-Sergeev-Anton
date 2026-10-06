# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · **Italiano**

Applicazione di ricerca su USD/RUB, EUR/RUB, CNY/RUB e GBP/RUB con dati della Banca di Russia. Progetto complementare del portfolio; non si ampliano le funzionalità.

## Funzioni e tecnologie

Python 3.10+, FastAPI, pandas, NumPy, LightGBM, XGBoost e scikit-learn; HTML/CSS/JS e Chart.js locale. Senza modello addestrato si usa una tendenza statistica. TF-IDF recupera documenti; il codice calcola i numeri e Ollama formula risposte aperte con sources. Redis ha un’alternativa nella memoria del processo. SQLite/SQLAlchemy registra assegnazione A/B e confronto delle previsioni. Sono presenti Prometheus, log JSON e Grafana opzionale.

## Limiti della valutazione

I test unitari/smoke non dimostrano vantaggio sulla persistenza (domani uguale a oggi), rendimenti o produzione. È presente valutazione walk-forward; servono dati riservati cronologicamente, hash, configurazione e risultati originali per confronti verificabili. Assegnazione A/B, MAE/MAPE o t-test non bastano: occorre considerare dipendenza temporale e osservazioni ripetute. Gli intervalli non provano da soli affidabilità né giustificano investimenti.

## Installazione, avvio e test

Configurare prima le variabili e i propri SECRET_KEY/JWT_SECRET_KEY secondo il modello del repository; non pubblicare segreti in Git. I comandi sono esempi, non prove di deployment. CI verifica Python 3.10/3.11; i test non dipendono dai servizi online della Banca di Russia o Ollama.

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

## API ed esempi

Le rotte per dati, previsioni, statistiche, assistente e salute sono sotto. /api/refresh, /api/force-refresh e /api/cache/status richiedono X-Admin-Key. L’assistente restituisce sources. /api/ab-test/* registra assegnazione/confronti; refresh-actuals deve aggiornare i tassi effettivi prima del confronto.

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

## Operatività e limiti

Compose pubblica la porta 8002 senza limitazione loopback predefinita; configurare l’accesso di rete prima del deployment. /monitoring/dashboard offre monitoraggio, /metrics Prometheus, /docs e /redoc documentazione API. Ollama e MLflow sono separati; MLFLOW_TRACKING_URI vuoto disattiva tracking. Il limite 768 MiB, un worker e l’obiettivo VDS 4 GiB non sono benchmark. Il profilo monitoring è opzionale. Verificare utente root di Compose, sicurezza, dipendenze, backup e HTTPS nel proprio ambiente.

## Lingue e licenza

Guide localizzate concise; esempi dettagliati in inglese o russo. Non aggiungono lingue a interfaccia, assistente o log. Consultare le condizioni originali di LICENSE; previsioni solo informative ed educative.

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
