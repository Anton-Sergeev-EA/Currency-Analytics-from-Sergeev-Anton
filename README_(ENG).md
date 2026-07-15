# Currency Analytics from Sergeev Anton.
Advanced currency analysis system with machine learning forecasting, RAG-based AI assistant, and interactive web 
interface.

## Description.
Currency Analytics from Sergeev Anton is a full-featured web application for analyzing and forecasting currency exchange rates using 
official data from the Central Bank of Russia. The system employs ensemble machine learning models (LightGBM, XGBoost, 
Random Forest, Gradient Boosting) to generate forecasts with confidence intervals. It includes a RAG-based AI assistant
for answering user questions and an interactive web interface with real-time charts.

## Features.
- Historical exchange rate data from CBR API (USD/RUB, EUR/RUB).
- Ensemble ML models for forecasting with uncertainty estimation.
- RAG-based AI assistant for currency-related queries.
- Interactive web interface with real-time charts and statistics.
- RESTful API with OpenAPI documentation (Swagger UI).
- Local caching system (Redis optional).
- Structured JSON logging.
- Docker containerization support.

## Technology Stack.
### Backend.
- Python 3.12 - Core language.
- FastAPI - Web framework.
- LightGBM, XGBoost, Random Forest, Gradient Boosting - ML models.
- Pandas, NumPy, Scikit-learn - Data processing.
- aiohttp - Asynchronous HTTP client.
- Jinja2 - HTML templating.

### Frontend.
- HTML5/CSS3 - Responsive layout.
- JavaScript (ES6) - Client-side logic.
- Chart.js - Interactive charts.

### Infrastructure.
- Redis - Optional caching.
- Docker - Containerization.
- Uvicorn - ASGI server.


## Installation.
### Prerequisites.
- Python 3.12 or higher.
- pip package manager.

### Local Installation.
1. Clone the repository:
   git clone <repository-url>
   cd currency-analyzer
2. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate  # Linux/Mac.
   venv\Scripts\activate     # Windows.
3. Install dependencies:
   pip install -r requirements.txt
4. Create environment configuration:
   cp .env.example .env
5. Train ML models:
   python train_models.py
6. Start the application:
   python -m src.main
7. Open in browser:
   http://localhost:8000/

### Docker Installation.
docker-compose up -d

## Usage.
### Web Interface.
Open in browser:
    http://localhost:8000/
The web interface provides:
- Real-time statistics - Current exchange rates and volatility.
- Interactive charts - Historical data visualization.
- Period selection - 30, 90, 180, 365 days.
- AI assistant - Chat interface for currency queries.
- Forecast table - 7-day currency forecasts.

## API Endpoints.
Method	     Endpoint	    Description
GET	         /	            Interactive web interface
GET	         /health	    System health check
GET	         /api/data	    Historical exchange rates
GET	         /api/forecast	Currency forecast
POST	     /api/ask	    RAG AI assistant query
GET	         /api/stats	    Data statistics
POST	     /api/refresh	Refresh cached data

## Interactive Documentation.
Open in browser:
http://localhost:8000/api/docs

## API Examples.
- Get forecast for USD:
curl "http://localhost:8000/api/forecast?days=7&currency=usd_rate"
- Ask AI assistant:
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the USD forecast for next week?"}'
- Get historical data:
curl "http://localhost:8000/api/data?period_days=30"
- Get statistics:
curl "http://localhost:8000/api/stats"

## Sample Questions for AI Assistant.
- "What is the USD forecast for next week?"
- "What is the EUR forecast?"
- "How much profit will I make investing 100,000 RUB in EUR?"
- "Compare USD and EUR"
- "What currency is better for investment?"

## Project Structure.
currency-analyzer/
├── src/
│   ├── core/              # Configuration, constants, exceptions.
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── exceptions.py
│   ├── domain/            # Domain entities.
│   │   ├── entities/
│   │   │   └── currency.py
│   │   └── value_objects/
│   │       └── money.py
│   ├── application/       # Business logic services.
│   │   └── services/
│   │       ├── data_service.py
│   │       ├── forecast_service.py
│   │       └── rag_service.py
│   ├── infrastructure/    # Data loading, ML models, RAG.
│   │   ├── data/
│   │   │   ├── cache.py
│   │   │   └── loader.py
│   │   ├── ml/
│   │   │   ├── features/
│   │   │   │   └── engineer.py
│   │   │   ├── models/
│   │   │   │   └── ensemble.py
│   │   │   ├── predictors/
│   │   │   │   └── predictor.py
│   │   │   └── trainers/
│   │   │       └── trainer.py
│   │   └── rag/
│   │       └── generation/
│   │           └── generator.py
│   ├── presentation/      # API routes, schemas, templates.
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── admin.py
│   │   │   │   ├── data.py
│   │   │   │   ├── forecast.py
│   │   │   │   ├── health.py
│   │   │   │   ├── rag.py
│   │   │   │   └── stats.py
│   │   │   └── schemas/
│   │   │       ├── request.py
│   │   │       └── response.py
│   │   └── templates/
│   │       └── index.html  # Web interface.
│   ├── common/            # Logging, utilities.
│   │   └── logger/
│   │       └── logger.py
│   └── main.py            # Application entry point.
├── data/                  # Data storage (created automatically).
│   ├── cache/
│   ├── models/
│   └── logs/
├── train_models.py        # Model training script.
├── requirements.txt       # Python dependencies.
├── Makefile               # Automation commands.
├── Dockerfile             # Docker configuration.
├── docker-compose.yml     # Docker Compose configuration.
├── .env                   # Environment variables.
├── .gitignore             # Git ignore rules.
├── LICENSE                # MIT License.
├── README.md              # Russian documentation.
├── README_ENG.md          # English documentation.
└── test_all.py            # API testing script.

## Dopment Commands.
make install       # Install dependencies.
make run           # Run application.
make test          # Run tests.
make clean         # Clean cache and temporary files.
make docker-build  # Build Docker image.
make docker-up     # Start Docker Compose services.
make docker-down   # Stop Docker Compose services.

## Testing.
### Run all tests:
python test_all.py
### Test specific endpoints
# Health check
curl http://localhost:8000/health
# Data
curl "http://localhost:8000/api/data?period_days=30"
# Forecast
curl "http://localhost:8000/api/forecast?days=7&currency=all"
# RAG
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How is the USD forecast?"}'
# Stats
curl http://localhost:8000/api/stats

# MIT License
Copyright (c) 2026 Sergeev Anton Valentinovich.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files, to deal in the Software without restriction, subject to the condition that the following 
attribution is clearly and prominently displayed in all copies, distributions, derivative works, and any other forms 
of the Software:

"Developed by Sergeev Anton Valentinovich (Anton V. Sergeev), 2026"

For full license terms, see the LICENSE file.

# Author.
Sergeev Anton Valentinovich
- Developer and maintainer.
- 2026.

# Disclaimer.
This software is provided for informational and educational purposes only. The author is not responsible for any 
financial decisions made based on the data or forecasts provided by this application. All investment decisions should 
be made after consultation with qualified financial advisors.

# Contributing.
1. Fork the repository.
2. Create a feature branch (git checkout -b feature/amazing-feature).
3. Commit changes (git commit -m 'Add amazing feature').
4. Push to branch (git push origin feature/amazing-feature).
5. Open a Pull Request.

# Acknowledgments.
- Central Bank of Russia for providing exchange rate data.
- Open-source community for the amazing libraries used in this project.
