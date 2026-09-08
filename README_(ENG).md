# Currency Analytics from Sergeev Anton.
Advanced currency analysis system with machine-learning-based forecasting, an AI assistant powered by RAG, and an interactive web interface.

## Description.
Currency Analytics from Sergeev Anton is a full-featured web application for analyzing and forecasting currency exchange rates using official data from the Central Bank of Russia. The system uses ensemble machine learning models (LightGBM, XGBoost, Random Forest, Gradient Boosting) to generate forecasts with confidence intervals. It includes an AI assistant based on RAG (Ollama + llama3.1:8b) for answering user questions, an interactive web interface with real-time charts, and a comprehensive monitoring dashboard for checking system health.

## Features.
- Historical currency exchange rate data from the Bank of Russia API (USD/RUB, EUR/RUB).
- Ensemble ML models for forecasting with uncertainty estimation.
- RAG-based AI assistant for answering questions about currencies.
- Interactive web interface with real-time charts and statistics.
- Monitoring dashboard displaying the status of all system components.
- Model performance tracking with accuracy metrics.
- A/B testing for comparing different model versions.
- RESTful API with OpenAPI documentation (Swagger UI).
- Redis-based caching.
- Structured JSON logging.
- Docker containerization with orchestration of multiple services.

## Technology Stack.
### Backend.
- Python 3.10
- FastAPI
- LightGBM, XGBoost, Random Forest, Gradient Boosting
- Pandas, NumPy, Scikit-learn
- aiohttp
- Jinja2
- SQLAlchemy
- Redis

### AI/ML.

- Ollama (llama3.1:8b) - RAG assistant.
- Sentence-Transformers - embeddings.
- ChromaDB - vector storage.

### Frontend.
- HTML5/CSS3 - Responsive layout.
- JavaScript (ES6) - Client-side logic.
- Vue.js - Reactive UI components.
- Chart.js - Interactive charts.

### Infrastructure.
- Redis - Caching and state management.
- Docker - Containerization.
- Docker Compose - Orchestration of multiple services.
- Uvicorn - ASGI server.

## Installation.
### System Requirements.
- Python 3.10 or higher.
- Docker and Docker Compose.
- Ollama.

### Quick Start with Docker.
1. Clone the repository:
git clone https://github.com/Anton-Sergeev-EA/Currency-Analytics-from-Sergeev-Anton.git
cd Currency-Analytics-from-Sergeev-Anton
2. Configure environment variables:
cp .env.example .env
3. Start all services:
docker-compose up -d
4. Check system health:
curl http://localhost:8000/health
```
5. Open in a browser:
- Main interface: http://localhost:8000
- Monitoring dashboard: http://localhost:8000/monitoring/dashboard
- API documentation: http://localhost:8000/docs
- Install Ollama (for RAG):
curl -fsSL https://ollama.com/install.sh | sh
- Download the model:
ollama pull llama3.1:8b
- Start Ollama:
ollama serve
```

### Local Installation (without Docker).
- Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac.
venv\Scripts\activate     # Windows.
```
- Install dependencies:
pip install -r requirements.txt
- Configure .env:
cp .env.example .env
```

- Start the application:
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
7. Open in a browser:
   - Main interface: http://localhost:8000
   - Monitoring dashboard: http://localhost:8000/monitoring/dashboard
   - API documentation: http://localhost:8000/docs

# API Endpoints.
| Method | Endpoint | Description |
|---|---|---|
| GET | / | Interactive web interface. |
| GET | /health | System health check. |
| GET | /api/data/data | Historical currency exchange rates. |
| GET | /api/forecast/forecast | Currency exchange rate forecast for 7-30 days. |
| POST | /api/rag/api/ask | Request to the AI assistant (RAG). |
| GET | /api/stats/stats | Data statistics. |
| GET | /ab-test/status | A/B testing status. |
| POST | /ab-test/predict | Forecast with A/B testing. |
| GET | /ab-test/stats | A/B test statistics. |
| GET | /monitoring/dashboard | Monitoring dashboard UI. |
| GET | /monitoring/api/health | Detailed component status. |
| GET | /monitoring/api/models | Information about loaded models. |
| GET | /monitoring/api/prediction-test | Model prediction tests. |
| GET | /monitoring/api/model-accuracy | Model accuracy metrics. |
| GET | /monitoring/api/current-rates | Current currency exchange rates. |
| GET | /monitoring/api/data-quality | Data quality metrics. |

## API Request Examples.
- Get a 7-day forecast for USD:
curl "http://localhost:8000/api/forecast/forecast?days=7&currency=USD"
```
- Request to the AI assistant:
curl -X POST http://localhost:8000/api/rag/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the forecast for the dollar next week?"}'
- Get historical data:
curl "http://localhost:8000/api/data/data?period_days=30"
```
- Get statistics:
curl "http://localhost:8000/api/stats"
```
- Check system health:
curl http://localhost:8000/health
```
- Test model predictions:
curl http://localhost:8000/monitoring/api/prediction-test
```
- A/B testing:
curl -X POST http://localhost:8000/ab-test/predict \
  -H "Content-Type: application/json" \
  -d '{"currency_pair": "USD/RUB", "forecast_days": 3, "user_id": "test_user"}'

## Example Questions for the AI Assistant.
- "What is the forecast for the dollar next week?"
- "What is the forecast for the euro?"
- "How much will I earn if I invest 100,000 rubles in euros?"
- "Compare the dollar and the euro."
- "Which currency is better for investments?"

# Project Management After Restart.
## Quick Start.
docker-compose up -d
## View Status.
docker-compose ps
## View Logs.
docker-compose logs -f currency-analytics
## Stop.
docker-compose down
## Full Rebuild.
docker-compose down
docker-compose up -d --build

## Project Structure.
Currency-Analytics-from-Sergeev-Anton/
├── src/
│   ├── ab_testing/       # A/B testing.
│   ├── application/      # Business logic services.
│   │   └── services/
│   ├── common/           # Logging, utilities.
│   │   └── logger/
│   ├── core/             # Configuration, constants.
│   ├── domain/           # Domain entities.
│   ├── infrastructure/   # Data, ML, RAG.
│   │   ├── data/         # Data loading and caching.
│   │   ├── ml/           # ML models and training.
│   │   └── rag/          # RAG components (Ollama).
│   ├── monitoring/       # Monitoring dashboard.
│   ├── presentation/     # API routes and templates.
│   └── main.py           # Entry point.
├── data/                 # Data (created automatically).
├── models/               # ML models.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README_RU.md
└── README_ENG.md
```

## Development Commands.
- `docker-compose up -d` - Start all services.
- `docker-compose down` - Stop all services.
- `docker-compose logs -f currency-analytics` - View logs.
- `docker-compose restart currency-analytics` - Restart the application.
- `docker-compose ps` - Container status.

## License.
MIT License.
Copyright (c) 2026 Sergeev Anton Valentinovich.
Permission is hereby granted to any person obtaining a copy of this software and associated documentation to use the Software without restriction, provided that the following information is included in all copies, distributions, derivative works, and any other forms of the Software:
"Developed by Sergeev Anton Valentinovich, 2026."
See the LICENSE file for the full license terms.

# Author.
1. Sergeev Anton Valentinovich.
2. Developer and maintainer.
3. 2026.

# MIT License.
Copyright (c) 2026, Sergeev Anton Valentinovich.
Permission is hereby granted to any person obtaining a copy of this software and associated documentation to use the Software without restriction, provided that the following information is included in all copies, distributions, derivative works, and any other forms of the Software:
"Developed by Sergeev Anton Valentinovich, 2026."
See the LICENSE file for the full license terms.

# Author.
Sergeev Anton Valentinovich  
Developer and maintainer.  
2026.

# Disclaimer.
This software is provided for informational and educational purposes. The author assumes no responsibility for financial decisions made based on the data or forecasts provided by this application. All investment decisions should be made after consulting qualified financial professionals.

# Acknowledgements.
- The Central Bank of Russia for providing currency exchange rate data.
- The Open Source community for excellent libraries.
- Ollama for an excellent platform for local LLMs.
