# Currency Analytics

English version: [README.md](README.md)

Веб-приложение для анализа и прогнозирования курсов USD/RUB и EUR/RUB на
основе официальных данных Банка России. Включает ансамблевую ML-модель
прогнозирования (LightGBM, XGBoost, Random Forest, Gradient Boosting),
RAG-ассистента на локальной модели Ollama, интерактивный дашборд и
демонстрационное A/B-тестирование.

## Возможности

- Исторические курсы валют из API ЦБ РФ (USD/RUB, EUR/RUB), с резервным
  источником и статистическим трендом на случай недоступности основного.
- Ансамблевое ML-прогнозирование с доверительными интервалами; при
  отсутствии обученной модели для валюты используется трендовый fallback.
- RAG-ассистент (Ollama, локально) для вопросов о курсах и прогнозах.
- Интерактивный веб-дашборд с графиками (Chart.js).
- Кэширование на Redis с автоматическим переключением на in-memory кэш,
  если Redis недоступен.
- Демонстрационное A/B-тестирование (`/api/ab-test/*`) — сейчас это
  фиксированные mock-ответы, а не реальное разделение трафика.
- Метрики Prometheus (`/metrics`) и профиль Compose для Grafana/Prometheus.
- Структурированное JSON-логирование.
- Набор smoke-тестов на pytest и CI на GitHub Actions.

## Технологии

**Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy
(только для лога A/B-теста), Redis, aiohttp.
**ML:** LightGBM, XGBoost, scikit-learn, pandas, NumPy.
**AI/RAG:** Ollama (локальная LLM), Sentence-Transformers, ChromaDB.
**Frontend:** обычные HTML/CSS/JS + Chart.js, без фреймворка.
**Инфраструктура:** Docker, Docker Compose, Prometheus, Grafana, nginx (прод).

## Быстрый старт (Docker)

```bash
git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
cd Currency-Analytics-from-Sergeev-Anton
cp .env.example .env

docker-compose up -d

curl http://localhost:8002/health
```

Далее откройте:
- Главный интерфейс: http://localhost:8002
- Дашборд мониторинга: http://localhost:8002/monitoring/dashboard
- Документация API (Swagger UI): http://localhost:8002/docs

Для ассистента отдельно установите и запустите [Ollama](https://ollama.com):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama
ollama serve
```

`tinyllama` — модель по умолчанию (`OLLAMA_MODEL` в `.env`); подойдёт
любая модель, поддерживаемая Ollama, более крупные просто отвечают медленнее.

## Локальная установка (без Docker)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env

uvicorn src.main:app --host 0.0.0.0 --port 8002
```

## Запуск тестов

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Тесты полностью офлайн (не требуют живых обращений к ЦБ РФ или Ollama) —
именно их запускает CI при каждом push. См. docstring модуля
`tests/test_app.py` о том, что проверяет каждый тест: часть из них —
регрессионные тесты для багов, из-за которых конкретные эндпоинты были
недоступны до этой ревизии.

## Справочник API

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | `/` | Веб-дашборд. |
| GET | `/health` | Человекочитаемая страница проверки здоровья. |
| GET | `/api/health` | Машиночитаемая проверка здоровья (JSON). |
| GET | `/api/ping` | Проверка доступности. |
| GET | `/api/data/data?period_days=N` | Исторические курсы валют. |
| GET | `/api/forecast/forecast?days=N&currency=USD\|EUR\|ALL` | ML-прогноз на 1-30 дней. |
| POST | `/api/rag/ask` | Задать вопрос ассистенту. |
| GET | `/api/stats/stats` | Текущие курсы и изменение за день. |
| POST | `/api/refresh` | Фоновое обновление данных (используется кнопкой "Обновить" на дашборде и cron-задачами `install.sh`). |
| POST | `/api/force-refresh` | Синхронное принудительное обновление данных, минуя кэш. |
| GET | `/api/cache/status` | Статус кэша. |
| GET | `/api/ab-test/status` \| `/predict` \| `/stats`, POST `/update-ratios` | Демонстрационные эндпоинты A/B-теста (mock-данные). |
| GET | `/monitoring/dashboard` | UI дашборда мониторинга. |
| GET | `/monitoring/api/health` \| `/models` \| `/current-rates` \| `/model-accuracy` \| `/prediction-test` \| `/data-quality` | Данные для дашборда. Часть из них (`model-accuracy`, `prediction-test`) возвращает случайные иллюстративные значения, а не результат реальной оценки модели — это отмечено прямо в коде. |
| GET | `/metrics` | Метрики Prometheus. |
| GET | `/docs`, `/redoc` | Документация OpenAPI. |

### Примеры

```bash
# Прогноз USD на 7 дней
curl "http://localhost:8002/api/forecast/forecast?days=7&currency=USD"

# Вопрос ассистенту
curl -X POST http://localhost:8002/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какой прогноз по доллару на следующую неделю?"}'

# Исторические данные
curl "http://localhost:8002/api/data/data?period_days=30"
```

### Примеры вопросов для ассистента

- "Какой прогноз по доллару на следующую неделю?"
- "Какой прогноз по евро?"
- "Сколько я заработаю, если вложу 100 000 рублей в евро?"
- "Сравни доллар и евро"
- "Какая валюта лучше для инвестиций?"

## Эксплуатация

```bash
make up            # docker-compose up -d
make deploy        # прод-конфигурация (docker-compose.prod.yml), с масштабированием
make logs-app      # логи приложения
make backup        # архивирование каталога data/ (включая обученные модели)
make test          # pytest tests/
```

`docker-compose.prod.yml` добавляет nginx и масштабирует приложение до 2
реплик за ним; `prometheus`/`grafana` — опциональный профиль Compose
`monitoring` (`docker-compose --profile monitoring up -d`).

## Лицензия

MIT с обязательным указанием авторства — см. [LICENSE](LICENSE).

## Отказ от ответственности

Предоставляется в информационных и образовательных целях. Прогнозы не
являются финансовой консультацией; перед принятием инвестиционных решений
на основе данных этого приложения проконсультируйтесь со специалистом.

## Благодарности

- Банку России — за данные о курсах валют.
- Проектам с открытым исходным кодом, на которых построено приложение.

---

Anton Sergeev — avsergeev1981@gmail.com
