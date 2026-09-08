import json
from typing import Dict, Any
from src.infrastructure.rag.generation.generator import ResponseGenerator
from src.application.services.forecast_service import ForecastService
from src.infrastructure.data.loader import DataLoader
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """Сервис обработки вопросов с привлечением RAG и Ollama."""

    def __init__(self):
        self.generator = ResponseGenerator()
        self.forecast_service = ForecastService()
        self.data_loader = DataLoader()

    async def ask(self, question: str) -> Dict[str, Any]:
        """Основной метод, вызываемый API роутером."""
        return await self.process_question(question)

    async def process_question(self, question: str) -> Dict[str, Any]:
        q_lower = question.lower().strip()

        greetings = ["привет", "здравствуй", "добрый день", "добрый вечер", "кто ты"]
        analysis_keywords = ["курс", "доллар", "евро", "usd", "eur", "прогноз", "сравни", "купить", "продать", "рубл", "динамик"]

        is_pure_greeting = any(g in q_lower for g in greetings) and not any(k in q_lower for k in analysis_keywords)

        if is_pure_greeting:
            return {
                "answer": (
                    "👋 Привет! Я Антон — ваш персональный финансовый аналитик.\n\n"
                    "Задайте любой вопрос по курсам валют (USD, EUR), прогнозам или их сравнению!"
                ),
                "type": "greeting"
            }

        try:
            # Загружаем свежие данные напрямую через DataLoader
            df = await self.data_loader.load_data()
            latest_usd = df["usd_rate"].iloc[-1] if df is not None and not df.empty and "usd_rate" in df.columns else "Н/Д"
            latest_eur = df["eur_rate"].iloc[-1] if df is not None and not df.empty and "eur_rate" in df.columns else "Н/Д"

            # Получаем 7-дневные прогнозы для обеих валют
            usd_forecast = await self.forecast_service.get_forecast(days=7, currency="usd_rate")
            eur_forecast = await self.forecast_service.get_forecast(days=7, currency="eur_rate")

            # Форматируем данные в JSON-строку для передачи в Ollama
            usd_str = json.dumps(usd_forecast, ensure_ascii=False) if isinstance(usd_forecast, (dict, list)) else str(usd_forecast)
            eur_str = json.dumps(eur_forecast, ensure_ascii=False) if isinstance(eur_forecast, (dict, list)) else str(eur_forecast)

            context = (
                f"=== ТЕКУЩИЕ КУРСЫ ЦБ РФ ===\n"
                f"USD/RUB: {latest_usd} руб.\n"
                f"EUR/RUB: {latest_eur} руб.\n\n"
                f"=== ПРОГНОЗ КУРСА USD/RUB (7 ДНЕЙ) ===\n{usd_str}\n\n"
                f"=== ПРОГНОЗ КУРСА EUR/RUB (7 ДНЕЙ) ===\n{eur_str}\n"
            )

            answer = await self.generator.generate_response(question, context)

            return {
                "answer": answer,
                "type": "ollama"
            }

        except Exception as e:
            logger.error(f"Error in RAGService processing question: {e}", exc_info=True)
            return {
                "answer": f"Произошла ошибка при анализе данных: {str(e)}",
                "type": "error"
            }
        