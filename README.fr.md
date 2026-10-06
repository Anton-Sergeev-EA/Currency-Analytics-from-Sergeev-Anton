# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md) · [Italiano](README.it.md)

Application de recherche sur USD/RUB, EUR/RUB, CNY/RUB et GBP/RUB avec les données de la Banque de Russie. Projet complémentaire du portfolio ; pas d’extension fonctionnelle prévue.

## Fonctions et technologies

Python 3.10+, FastAPI, pandas, NumPy, LightGBM, XGBoost et scikit-learn ; HTML/CSS/JS et Chart.js local. Sans modèle entraîné, repli sur une tendance statistique. TF-IDF recherche les documents ; le code calcule les nombres et Ollama rédige les réponses ouvertes avec sources. Redis dispose d’un repli en mémoire du processus. SQLite/SQLAlchemy journalise affectation A/B et comparaison des prévisions. Prometheus, logs JSON et Grafana facultatif sont disponibles.

## Limites de l’évaluation

Les tests unitaires/smoke ne prouvent ni avantage sur la persistance (demain égale aujourd’hui), ni rendement, ni déploiement en production. Le code comprend une évaluation walk-forward ; il faut données réservées chronologiquement, hashes, configuration et résultats bruts pour une comparaison vérifiable. Affectation A/B, MAE/MAPE ou t-test ne suffisent pas : dépendance temporelle et observations répétées comptent. Les intervalles ne démontrent pas seuls la fiabilité et ne justifient pas des investissements.

## Installation, lancement et tests

Configurez auparavant les variables et vos propres SECRET_KEY/JWT_SECRET_KEY selon le modèle du dépôt ; ne publiez pas les secrets dans Git. Les commandes sont des exemples, pas des preuves de déploiement. CI teste Python 3.10/3.11 ; les tests ne dépendent pas des services en ligne de la Banque de Russie ou d’Ollama.

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

## API et exemples

Les routes données, prévisions, statistiques, assistant et santé figurent ci-dessous. /api/refresh, /api/force-refresh et /api/cache/status exigent X-Admin-Key. L’assistant renvoie sources. /api/ab-test/* journalise affectation et comparaison ; refresh-actuals doit mettre à jour les taux réels avant comparaison.

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

## Exploitation et limites

Compose publie le port 8002 sans restriction loopback par défaut ; configurez l’accès réseau avant déploiement. /monitoring/dashboard fournit la supervision, /metrics Prometheus, /docs et /redoc la documentation API. Ollama et MLflow sont séparés ; MLFLOW_TRACKING_URI vide désactive le suivi. La limite 768 MiB, un worker et la cible VDS 4 GiB ne sont pas des mesures de performance. Le profil monitoring est facultatif. Vérifiez utilisateur root dans Compose, sécurité, dépendances, sauvegardes et HTTPS dans votre environnement.

## Langues et licence

Guides localisés concis ; exemples détaillés en anglais ou russe. Ils n’ajoutent pas de langues à l’interface, à l’assistant ou aux logs. Consultez les conditions originales de LICENSE ; prévisions uniquement informatives et éducatives.

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
