# Currency Analytics from Sergeev Anton.
Продвинутая система анализа валют с прогнозированием на основе машинного обучения, AI-ассистентом на базе RAG и интерактивным веб-интерфейсом.

## Описание.
Currency Analytics from Sergeev Anton - это полнофункциональное веб-приложение для анализа и прогнозирования курсов валют с использованием официальных данных Центрального Банка России. Система применяет ансамблевые модели машинного обучения (LightGBM, XGBoost, Random Forest, Gradient Boosting) для формирования прогнозов с доверительными интервалами. Включает AI-ассистента на основе RAG (Ollama + llama3.1:8b) для ответов на вопросы пользователей, интерактивный веб-интерфейс с графиками в реальном времени и комплексный дашборд мониторинга для проверки работоспособности системы.

## Возможности.
- Исторические данные курсов валют из API ЦБ РФ (USD/RUB, EUR/RUB).
- Ансамблевые ML-модели для прогнозирования с оценкой неопределенности.
- AI-ассистент на основе RAG для ответов на вопросы о валютах.
- Интерактивный веб-интерфейс с графиками и статистикой в реальном времени.
- Дашборд мониторинга с отображением статуса всех компонентов системы.
- Отслеживание производительности моделей с метриками точности.
- A/B тестирование для сравнения разных версий моделей.
- RESTful API с документацией OpenAPI (Swagger UI).
- Кэширование на базе Redis.
- Структурированное JSON-логирование.
- Docker-контейнеризация с оркестрацией нескольких сервисов.

## Технологический стек.
### Бэкенд.
- Python 3.10
- FastAPI
- LightGBM, XGBoost, Random Forest, Gradient Boosting 
- Pandas, NumPy, Scikit-learn
- aiohttp
- Jinja2
- SQLAlchemy
- Redis

### AI/ML.
- Ollama (llama3.1:8b) - RAG ассистент.
- Sentence-Transformers - эмбеддинги.
- ChromaDB - векторное хранилище.

### Фронтенд.
- HTML5/CSS3 - Адаптивная верстка.
- JavaScript (ES6) - Клиентская логика.
- Vue.js - Реактивные компоненты UI.
- Chart.js - Интерактивные графики.

### Инфраструктура.
- Redis - Кэширование и управление состоянием.
- Docker - Контейнеризация.
- Docker Compose - Оркестрация нескольких сервисов.
- Uvicorn - ASGI-сервер.

## Установка.
### Системные требования.
- Python 3.10 или выше.
- Docker и Docker Compose.
- Ollama.

### Быстрый старт через Docker.
1. Клонирование репозитория:
git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
cd Currency-Analytics-from-Sergeev-Anton
2. Настройка переменных окружения:
cp .env.example .env
3. Запуск всех сервисов:
docker-compose up -d
4. Проверка работоспособности:
curl http://localhost:8000/health
5. Открытие в браузере:
- Главный интерфейс: http://localhost:8000
- Дашборд мониторинга: http://localhost:8000/monitoring/dashboard
- Документация API: http://localhost:8000/docs

- Установка Ollama (для RAG).
curl -fsSL https://ollama.com/install.sh | sh
- Скачивание модели.
ollama pull llama3.1:8b
- Запуск Ollama.
ollama serve

### Локальная установка (без Docker).
- Создание виртуального окружения.
python -m venv venv
source venv/bin/activate  # Linux/Mac.
venv\Scripts\activate     # Windows.
- Установка зависимостей.
pip install -r requirements.txt
- Настройка .env.
cp .env.example .env
- Запуск приложения.
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

7. Открытие в браузере:
   - Основной интерфейс: http://localhost:8000
   - Дашборд мониторинга: http://localhost:8000/monitoring/dashboard
   - Документация API: http://localhost:8000/docs

# API Эндпоинты.
Метод	Эндпоинт	                         Описание
GET	/	                               Интерактивный веб-интерфейс.
GET	/health	                         Проверка работоспособности системы.
GET	/api/data/data	                   Исторические курсы валют.
GET	/api/forecast/forecast	          Прогноз курсов валют на 7-30 дней.
POST	/api/rag/api/ask	                Запрос к AI-ассистенту (RAG).
GET	/api/stats/stats	                Статистика по данным.
GET	/ab-test/status	                Статус A/B тестирования.
POST	/ab-test/predict	                Прогноз с A/B тестированием.
GET	/ab-test/stats	                   Статистика A/B теста.
GET	/monitoring/dashboard	          UI дашборда мониторинга.
GET	/monitoring/api/health	          Детальный статус компонентов.
GET	/monitoring/api/models	          Информация о загруженных моделях.
GET	/monitoring/api/prediction-test	 Тестовые предсказания моделей.
GET	/monitoring/api/model-accuracy	 Метрики точности моделей.
GET	/monitoring/api/current-rates	    Текущие курсы валют.
GET	/monitoring/api/data-quality	    Метрики качества данных.

## Примеры API-запросов.
- Получение прогноза для USD на 7 дней:
curl "http://localhost:8000/api/forecast/forecast?days=7&currency=USD"
- Запрос к AI-ассистенту:
curl -X POST http://localhost:8000/api/rag/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какой прогноз по доллару на следующую неделю?"}'
- Получение исторических данных:
curl "http://localhost:8000/api/data/data?period_days=30"
- Получение статистики:
curl "http://localhost:8000/api/stats"
- Проверка здоровья системы:
curl http://localhost:8000/health
- Тестирование предсказаний моделей:
curl http://localhost:8000/monitoring/api/prediction-test
- A/B тестирование:
curl -X POST http://localhost:8000/ab-test/predict \
  -H "Content-Type: application/json" \
  -d '{"currency_pair": "USD/RUB", "forecast_days": 3, "user_id": "test_user"}'

## Примеры вопросов для AI-ассистента.
- "Какой прогноз по доллару на следующую неделю?"
- "Какой прогноз по евро?"
- "Сколько я заработаю, если вложу 100 000 рублей в евро?"
- "Сравни доллар и евро"
- "Какая валюта лучше для инвестиций?"

# Управление проектом после перезагрузки
## Быстрый запуск.
docker-compose up -d
## Просмотр статуса.
docker-compose ps
## Просмотр логов.
docker-compose logs -f currency-analytics
## Остановка.
docker-compose down
## Полная пересборка.
docker-compose down
docker-compose up -d --build

## Структура проекта.
Currency-Analytics-from-Sergeev-Anton/
├── src/
│   ├── ab_testing/       # A/B тестирование.
│   ├── application/      # Сервисы бизнес-логики.
│   │   └── services/
│   ├── common/           # Логирование, утилиты.
│   │   └── logger/
│   ├── core/             # Конфигурация, константы.
│   ├── domain/           # Доменные сущности.
│   ├── infrastructure/   # Данные, ML, RAG.
│   │   ├── data/         # Загрузка и кэширование данных.
│   │   ├── ml/           # ML модели и обучение.
│   │   └── rag/          # RAG компоненты (Ollama).
│   ├── monitoring/       # Дашборд мониторинга.
│   ├── presentation/     # API маршруты и шаблоны.
│   └── main.py           # Точка входа.
├── data/                 # Данные (создается автоматически).
├── models/               # ML модели.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README_RU.md
└── README_ENG.md

## Команды для разработки.
- docker-compose up -d - Запуск всех сервисов.
- docker-compose down - Остановка всех сервисов.
- docker-compose logs -f currency-analytics - Просмотр логов.
- docker-compose restart currency-analytics - Перезапуск приложения.
- docker-compose ps - Статус контейнеров.

## Лицензия.
MIT License.
Copyright (c) 2026 Сергеев Антон Валентинович.
Настоящим предоставляется разрешение любому лицу, получающему копию данного программного обеспечения и сопутствующей 
документации, использовать Программное Обеспечение без ограничений, при условии обязательного указания следующей 
информации во всех копиях, распространениях, производных работах и любых других формах Программного Обеспечения:

"Разработано Сергеевым Антоном Валентиновичем, 2026 г."

Полные условия лицензии см. в файле LICENSE.

# Автор.
1. Сергеев Антон Валентинович.
2. Разработчик и сопровождающий.
3. 2026 г.

# MIT License.
Copyright (c) 2026, Сергеев Антон Валентинович.
Настоящим предоставляется разрешение любому лицу, получающему копию данного программного обеспечения и сопутствующей документации, использовать Программное Обеспечение без ограничений, при условии обязательного указания следующей информации во всех копиях, распространениях, производных работах и любых других формах Программного Обеспечения:

"Разработано Сергеевым Антоном Валентиновичем, 2026 г."
Полные условия лицензии см. в файле LICENSE.

# Автор.
Сергеев Антон Валентинович
Разработчик и сопровождающий.
2026 г.

# Отказ от ответственности.
Данное программное обеспечение предоставляется в информационных и образовательных целях. Автор не несет ответственности за финансовые решения, принятые на основе данных или прогнозов, предоставляемых данным приложением. Все инвестиционные решения должны приниматься после консультации с квалифицированными финансовыми специалистами.

# Благодарности.
- Центральный Банк России за предоставление данных о курсах валют.
- Сообщество Open Source за замечательные библиотеки.
- Ollama за отличную платформу для локальных LLM.
