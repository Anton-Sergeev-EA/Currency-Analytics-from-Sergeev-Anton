# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch** · [Italiano](README.it.md)

Forschungsanwendung für USD/RUB, EUR/RUB, CNY/RUB und GBP/RUB mit Daten der Bank von Russland. Ergänzendes Portfolio-Projekt; keine funktionale Erweiterung vorgesehen.

## Funktionen und Technik

Python 3.10+, FastAPI, pandas, NumPy, LightGBM, XGBoost und scikit-learn; HTML/CSS/JS und lokales Chart.js. Ohne trainiertes Modell wird ein statistischer Trend verwendet. TF-IDF sucht Dokumente; Code berechnet Zahlen, Ollama formuliert offene Antworten mit sources. Redis hat einen Ersatz im Prozessspeicher. SQLite/SQLAlchemy protokolliert A/B-Zuweisung und Prognosevergleiche. Prometheus, JSON-Logs und optionales Grafana sind vorhanden.

## Bewertungsgrenzen

Unit-/Smoke-Tests belegen weder einen Vorteil gegenüber Persistenz (morgen gleich heute) noch Handelsrendite oder Produktionsbetrieb. Walk-forward-Auswertung ist implementiert; nachprüfbare Vergleiche benötigen chronologisch zurückgehaltene Daten, Hashes, Konfiguration und Rohresultate. A/B-Zuweisung, MAE/MAPE oder t-Test allein reichen nicht: Zeitabhängigkeit und wiederholte Beobachtungen müssen berücksichtigt werden. Intervalle allein belegen keine Zuverlässigkeit und rechtfertigen keine Anlageentscheidung.

## Installation, Start und Tests

Zuerst Umgebungsvariablen und eigene SECRET_KEY/JWT_SECRET_KEY gemäß Repository-Vorlage konfigurieren; Geheimnisse nicht in Git veröffentlichen. Befehle sind Beispiele, keine Deployment-Nachweise. CI testet Python 3.10/3.11; Tests benötigen keine Online-Dienste der Bank von Russland oder Ollama.

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

## API und Beispiele

Routen für Daten, Prognosen, Statistik, Assistent und Status stehen unten. /api/refresh, /api/force-refresh und /api/cache/status benötigen X-Admin-Key. Der Assistent liefert sources. /api/ab-test/* protokolliert Zuweisung/Vergleiche; refresh-actuals muss vor Vergleichen reale Kurse nachtragen.

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

## Betrieb und Grenzen

Compose veröffentlicht Port 8002 standardmäßig ohne Loopback-Beschränkung; Netzwerkzugriff vor Deployment konfigurieren. /monitoring/dashboard bietet Überwachung, /metrics Prometheus, /docs und /redoc API-Dokumentation. Ollama und MLflow sind separate Dienste; leeres MLFLOW_TRACKING_URI deaktiviert Tracking. 768 MiB Limit, ein Worker und das 4-GiB-VDS-Ziel sind keine Benchmarks. Das Profil monitoring ist optional. Root-Benutzer in Compose, Sicherheit, Abhängigkeiten, Backups und HTTPS im eigenen Umfeld prüfen.

## Sprachen und Lizenz

Kompakte übersetzte Anleitungen; ausführliche Beispiele auf Englisch oder Russisch. Keine neue Sprachunterstützung für Oberfläche, Assistent oder Logs. Originalbedingungen in LICENSE beachten; Prognosen dienen nur Information und Bildung.

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
