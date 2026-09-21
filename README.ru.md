# Currency Analytics

English version: [README.md](README.md)

Веб-приложение для анализа и прогнозирования курсов USD/RUB и EUR/RUB на
основе официальных данных Банка России. Включает ансамблевую ML-модель
прогнозирования (LightGBM, XGBoost, Random Forest, Gradient Boosting),
RAG-ассистента на локальной модели Ollama, интерактивный дашборд и
настоящее A/B-тестирование модели против статистического baseline.

Проект специально спроектирован так, чтобы работать на слабом VDS
(4 ГБ ОЗУ, немного места на диске) — см. раздел [«Деплой на слабый
VDS»](#деплой-на-слабый-vds).

## Возможности

- Исторические курсы валют из API ЦБ РФ (USD/RUB, EUR/RUB), с резервным
  источником и статистическим трендом на случай недоступности основного.
- Ансамблевое ML-прогнозирование с доверительными интервалами; при
  отсутствии обученной модели для валюты используется трендовый fallback.
- **RAG-ассистент с настоящим retrieval**: TF-IDF-поиск по базе знаний
  (без тяжёлых torch/sentence-transformers/chromadb — это лёгкий и
  проверяемый метод для небольшого корпуса документов), плюс
  **детерминированные финансовые расчёты** (конвертация, инвестиционный
  прогноз, сравнение валют) — цифры считаются кодом, а не
  "угадываются" маленькой локальной LLM. Ollama используется только там,
  где она сильна: формулировка ответа на открытые вопросы по найденному
  контексту. В ответе API возвращается `sources` — какие документы базы
  знаний реально использовались.
- Интерактивный веб-дашборд с графиками (Chart.js), конвертером валют и
  тёмной темой.
- Кэширование на Redis с автоматическим переключением на in-memory кэш,
  если Redis недоступен.
- **Настоящее A/B-тестирование** (`/api/ab-test/*`): вариант A — реальная
  ансамблевая ML-модель, вариант B — прогноз "без изменений"
  (persistence/random-walk — стандартный бенчмарк в прогнозировании
  валютных курсов). Детерминированное распределение пользователей по
  вариантам, лог в SQLite, реальные MAE/MAPE и t-тест статистической
  значимости после того, как фактический курс становится известен
  (`POST /api/ab-test/refresh-actuals`).
- **Настоящие метрики на дашборде мониторинга**: точность модели
  считается честным walk-forward бэктестом (обучение без последних N
  дней, проверка на них), а не случайными числами; качество данных и
  прогноз на дашборде — реальные, посчитанные из фактических данных.
- Метрики Prometheus (`/metrics`) и опциональный профиль Compose для
  Grafana/Prometheus.
- Структурированное JSON-логирование.
- Набор smoke-тестов на pytest и CI на GitHub Actions.

## Технологии

**Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy
(лог A/B-теста), Redis, aiohttp.
**ML:** LightGBM, XGBoost, scikit-learn, pandas, NumPy.
**AI/RAG:** Ollama (локальная LLM), TF-IDF-ретривер на scikit-learn
(намеренно без torch/sentence-transformers/chromadb — см. ниже).
**Frontend:** обычные HTML/CSS/JS + Chart.js, без фреймворка.
**Инфраструктура:** Docker (multi-stage сборка), Docker Compose,
Prometheus, Grafana (опционально), nginx (прод).

## Деплой на слабый VDS

Проект по умолчанию настроен под сервер с **4 ГБ ОЗУ и небольшим диском**:

- Тяжёлые ML-библиотеки для эмбеддингов (torch/sentence-transformers/
  chromadb) намеренно не используются — вместо них лёгкий TF-IDF-поиск
  на scikit-learn (уже нужен для прогнозов). Это экономит ~1.5–2 ГБ
  места на диске и сотни МБ ОЗУ.
- Docker-образ собирается в 2 стадии (`Dockerfile`): компиляторы
  (gcc/g++/make) остаются только в промежуточном слое сборки и не
  попадают в финальный образ.
- По умолчанию **1 воркер uvicorn** (`UVICORN_WORKERS=1` в
  `docker-compose.yml`) — каждый дополнительный воркер это ещё одна
  полная копия pandas/numpy/lightgbm/xgboost в памяти.
- Жёсткий лимит памяти на контейнер приложения — 768 МБ
  (`mem_limit` в `docker-compose.yml`).
- В `docker-compose.prod.yml` по умолчанию **1 реплика** приложения
  (`REPLICAS=1`) вместо прежних 2 — поднимайте больше, только если
  `free -h` показывает реальный запас.
- Для ассистента используйте `tinyllama` или другую модель на 1–2B
  параметров — более крупные будут конкурировать с самим приложением
  за ту же память.
- Стек Prometheus/Grafana полностью опционален (`--profile monitoring`)
  и не нужен для работы приложения — на маленьком VDS чаще достаточно
  встроенного дашборда `/monitoring/dashboard`.
- Рекомендуется настроить **swap** на VDS (2 ГБ+), если ещё не
  настроен — на случай кратковременных пиков (например, инференс
  Ollama во время обучения модели).

## Деплой на домен anton-analytics.ru с HTTPS

Проект настроен на публикацию по адресу **https://anton-analytics.ru**.
Ниже описан обычный случай: **общий VDS**, на котором уже работает свой
nginx на хосте и уже есть действующий сертификат Let's Encrypt для домена
(например, потому что на этом же сервере живёт другой проект или
предыдущая версия приложения) - а не развёртывание HTTPS с нуля.

1. **Клонируйте репозиторий и запустите стек на отдельном порту:**

   ```bash
   git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
   cd Currency-Analytics-from-Sergeev-Anton
   cp .env.example .env   # обязательно проверьте SECRET_KEY/JWT_SECRET_KEY
   docker compose up -d --build
   ```

   Приложение слушает `127.0.0.1:8002` (см. `docker-compose.yml`),
   поэтому не конкурирует за порты 80/443 с тем, что уже работает на
   сервере.

2. **Перенаправьте существующий nginx-сайт на него.** В конфиге сайта
   (например, `/etc/nginx/sites-available/anton-analytics`) поменяйте
   `proxy_pass` на `http://127.0.0.1:8002`, затем:

   ```bash
   sudo nginx -t && sudo systemctl reload nginx
   ```

   Запускать certbot не нужно - сертификат уже покрывает домен.

3. **Проверьте и только потом выключайте старую версию:**

   ```bash
   curl -I https://anton-analytics.ru/health
   ```

   Откройте https://anton-analytics.ru и
   https://anton-analytics.ru/monitoring/dashboard, убедитесь, что всё
   работает, и только после этого останавливайте сервис, который раньше
   отвечал на этом домене.

### Автозапуск и восстановление после перезагрузки

- Контейнер `currency-analytics` запущен с `restart: unless-stopped`
  (`docker-compose.yml`) и `healthcheck` (`GET /health` каждые 30
  секунд) - Docker перезапускает его при падении и помечает нездоровым
  при сбое.
- `systemctl enable docker` (обычно уже включено, если на сервере и так
  крутятся другие Docker-проекты) гарантирует, что демон - и все
  контейнеры с `restart: unless-stopped` - поднимутся после
  перезагрузки.
- CORS управляется переменной `ALLOWED_ORIGINS` в `.env`/
  `docker-compose.yml` (по умолчанию - продакшн-домен + локальная
  разработка), а не захардкоженной `*`.

Полезные команды:

```bash
docker compose ps
docker compose logs -f
sudo nginx -T | grep -A5 anton-analytics   # проверить, на какой порт проксирует nginx
```

> Развёртывание HTTPS на чистом сервере без существующего nginx и
> сертификата - другой, более редкий сценарий; `docker-compose.prod.yml`
> и `scripts/init_letsencrypt.sh` остаются в репозитории именно для
> него, но не используются в описанном выше деплое anton-analytics.ru.

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

`tinyllama` — модель по умолчанию (`OLLAMA_MODEL` в `.env`) и
рекомендуемая для слабого VDS; более крупные модели работают точнее, но
требуют больше ОЗУ и отвечают медленнее.

## Локальная установка (без Docker)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env

uvicorn src.main:app --host 0.0.0.0 --port 8002
```

Чтобы прогнозы использовали настоящую ML-модель, а не трендовый
fallback, обучите модели один раз на исторических данных:

```bash
python train_models.py
```

## Запуск тестов

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Тесты полностью офлайн (не требуют живых обращений к ЦБ РФ или Ollama) —
именно их запускает CI при каждом push.

## Справочник API

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | `/` | Веб-дашборд. |
| GET | `/health` | Человекочитаемая страница проверки здоровья. |
| GET | `/api/health` | Машиночитаемая проверка здоровья (JSON). |
| GET | `/api/ping` | Проверка доступности. |
| GET | `/api/data/data?period_days=N` | Исторические курсы валют. |
| GET | `/api/forecast/forecast?days=N&currency=USD\|EUR\|ALL` | ML-прогноз на 1-30 дней. |
| POST | `/api/rag/ask` | Задать вопрос ассистенту (ответ включает `sources` — использованные документы базы знаний). |
| GET | `/api/stats/stats` | Текущие курсы и изменение за день. |
| POST | `/api/refresh` | Фоновое обновление данных. |
| POST | `/api/force-refresh` | Синхронное принудительное обновление данных, минуя кэш. |
| GET | `/api/cache/status` | Статус кэша. |
| GET | `/api/ab-test/status` | Статус A/B-теста и текущее распределение трафика. |
| POST | `/api/ab-test/predict?currency=usd_rate&days=1` | Реальный прогноз через назначенный вариант (A/B), логируется в БД. |
| GET | `/api/ab-test/stats?days=30` | Реальные MAE/MAPE по вариантам + t-тест значимости. |
| POST | `/api/ab-test/update-ratios?split_a=0.5` | Изменить долю трафика на вариант A. |
| POST | `/api/ab-test/refresh-actuals` | Подтянуть фактические курсы для прогнозов с наступившей датой (нужно для расчёта метрик). |
| GET | `/monitoring/dashboard` | UI дашборда мониторинга. |
| GET | `/monitoring/api/health` \| `/models` \| `/current-rates` \| `/model-accuracy` \| `/prediction-test` \| `/data-quality` | Данные для дашборда — все посчитаны из реальных данных/моделей (см. `ModelEvaluator`), с кэшированием на несколько часов для тяжёлых метрик. |
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

# Реальный расчёт: сколько заработаю
curl -X POST http://localhost:8002/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько я заработаю, если вложу 100000 рублей в евро на месяц?"}'

# Исторические данные
curl "http://localhost:8002/api/data/data?period_days=30"
```

### Примеры вопросов для ассистента

- «Какой прогноз по доллару на следующую неделю?» — реальный расчёт из ML-прогноза.
- «Сколько я заработаю, если вложу 100 000 рублей в евро на месяц?» — точный расчёт, не догадка LLM.
- «Переведи 500 долларов в рубли» — конвертация по текущему курсу ЦБ.
- «Сравни доллар и евро» — сравнение прогнозируемой динамики.
- «Какая валюта лучше для инвестиций?» — открытый вопрос, ответ строится через RAG-поиск по базе знаний + Ollama.

## Эксплуатация

```bash
make up            # docker-compose up -d
make deploy        # прод-конфигурация (docker-compose.prod.yml)
make logs-app      # логи приложения
make backup        # архивирование каталога data/ (включая обученные модели)
make test          # pytest tests/
```

Для обновления фактических курсов A/B-теста и статистики точности
рекомендуется добавить в cron (см. `cron_setup.txt`):

```bash
python scripts/update_ab_actual_rates.py
```

`docker-compose.prod.yml` добавляет nginx перед приложением;
`prometheus`/`grafana` — опциональный профиль Compose `monitoring`
(`docker-compose --profile monitoring up -d`), не обязателен для работы.

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
