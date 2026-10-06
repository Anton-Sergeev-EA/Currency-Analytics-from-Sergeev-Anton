# Currency Analytics

[Русский](README.ru.md) · [English](README.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

Aplicación de investigación para analizar USD/RUB, EUR/RUB, CNY/RUB y GBP/RUB con datos del Banco de Rusia. Proyecto complementario del portafolio; no se amplían funciones.

## Funciones y tecnología

Python 3.10+, FastAPI, pandas, NumPy, LightGBM, XGBoost y scikit-learn; HTML/CSS/JS y Chart.js local. Sin modelo entrenado se usa una tendencia estadística. TF-IDF recupera documentos; el código calcula números y Ollama redacta respuestas abiertas con sources. Redis tiene alternativa en memoria del proceso. SQLite/SQLAlchemy registra asignación A/B y comparación de previsiones. Incluye Prometheus, registros JSON y Grafana opcional.

## Límites de evaluación

Los tests unitarios/smoke no demuestran ventaja sobre persistencia (mañana igual a hoy), rentabilidad ni despliegue en producción. Hay evaluación walk-forward; una comparación verificable requiere datos reservados cronológicamente, hashes, configuración y resultados originales. Asignación A/B, MAE/MAPE o t-test no bastan: deben considerarse dependencia temporal y observaciones repetidas. Los intervalos no prueban por sí solos fiabilidad ni justifican decisiones de inversión.

## Instalación, ejecución y pruebas

Configure antes las variables de entorno y sus propios SECRET_KEY/JWT_SECRET_KEY según la plantilla del repositorio; no publique secretos en Git. Son ejemplos, no evidencia de despliegue. CI prueba Python 3.10/3.11; los tests no dependen del Banco de Rusia ni Ollama en línea.

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

## API y ejemplos

Las rutas de datos, previsiones, estadísticas, asistente y salud están abajo. /api/refresh, /api/force-refresh y /api/cache/status requieren X-Admin-Key. El asistente devuelve sources. /api/ab-test/* registra asignación/comparaciones; refresh-actuals debe actualizar valores reales antes de comparar.

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

## Operación y limitaciones

Compose publica el puerto 8002 sin restringirlo por defecto a loopback; ajuste el acceso de red antes de desplegar. /monitoring/dashboard ofrece monitorización, /metrics Prometheus y /docs y /redoc documentación API. Ollama y MLflow son servicios separados; MLFLOW_TRACKING_URI vacío desactiva seguimiento. El límite de 768 MiB, un worker y el objetivo VDS de 4 GiB no son benchmarks. El perfil monitoring es opcional. Revise el usuario root de Compose, seguridad, dependencias, copias y HTTPS en su entorno.

## Idiomas y licencia

Guías localizadas breves; ejemplos completos en inglés o ruso. No añaden idiomas a la interfaz, al asistente ni a los registros. Consulte las condiciones originales de LICENSE; previsiones solo informativas y educativas.

[English](README.md) · [Русский](README.ru.md) · [LICENSE](LICENSE)
