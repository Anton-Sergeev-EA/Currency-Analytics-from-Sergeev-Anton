import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.application.services.data_service import DataService
from src.application.services.forecast_service import ForecastService

logger = logging.getLogger(__name__)


class RAGGenerator:
    def __init__(
        self,
        data_service: Optional[DataService] = None,
        forecast_service: Optional[ForecastService] = None
    ):
        self.data_service = data_service or DataService()
        self.forecast_service = forecast_service or ForecastService()

    async def generate_response(self, question: str, context: str = "") -> str:
        """
        Генерирует умный ответ на основе вопроса пользователя.
        """
        question_lower = question.lower()
        
        try:
            if "прогноз" in question_lower or "forecast" in question_lower:
                return await self._handle_forecast(question_lower)
            else:
                return await self._handle_general(question_lower)
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    async def _handle_forecast(self, question: str) -> str:
        """
        Обрабатывает запросы прогноза с учетом конкретной валюты.
        """
        try:
            forecast_data = await self.forecast_service.get_forecast_with_uncertainty(7)
            
            usd_means = forecast_data.get("usd_rate", {}).get("mean", [])
            eur_means = forecast_data.get("eur_rate", {}).get("mean", [])
            
            usd_next = usd_means[0] if usd_means else 0.0
            eur_next = eur_means[0] if eur_means else 0.0
            
            ask_eur = "евро" in question or "eur" in question
            ask_usd = "доллар" in question or "usd" in question
            
            if ask_eur and not ask_usd:
                return (
                    f"Прогноз курса Евро (EUR) на ближайшие дни:\n"
                    f"- Завтра: около {eur_next:.2f} руб.\n"
                    f"- Тренд на неделю: ожидается в диапазоне около {eur_next:.2f} — {(eur_means[-1] if eur_means else eur_next):.2f} руб."
                )
            elif ask_usd and not ask_eur:
                return (
                    f"Прогноз курса Доллар США (USD) на ближайшие дни:\n"
                    f"- Завтра: около {usd_next:.2f} руб.\n"
                    f"- Тренд на неделю: ожидается в диапазоне около {usd_next:.2f} — {(usd_means[-1] if usd_means else usd_next):.2f} руб."
                )
            else:
                return (
                    f"Прогноз курсов валют на ближайшее время:\n"
                    f"- Доллар США (USD): около {usd_next:.2f} руб.\n"
                    f"- Евро (EUR): около {eur_next:.2f} руб."
                )
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return "Не удалось сформировать прогноз из-за ошибки в данных."

    async def _handle_general(self, question: str) -> str:
        """
        Обрабатывает общие вопросы (инвестиции, текущие курсы и т.д.).
        """
        try:
            df = await self.data_service.get_historical_data(7)
            
            if df is not None and not df.empty:
                usd_curr = float(df['usd_rate'].iloc[-1])
                eur_curr = float(df['eur_rate'].iloc[-1])
                
                if "инвестиц" in question or "лучше" in question:
                    return (
                        f"Для инвестиций стоит учитывать текущую волатильность.\n"
                        f"Актуальные котировки:\n"
                        f"- USD: {usd_curr:.4f} руб.\n"
                        f"- EUR: {eur_curr:.4f} руб.\n"
                        f"Диверсификация портфеля между обеими валютами исторически снижает риски."
                    )
                
                return (
                    f"Актуальные официальные курсы валют ЦБ РФ:\n"
                    f"- Доллар США (USD): {usd_curr:.4f} руб.\n"
                    f"- Евро (EUR): {eur_curr:.4f} руб."
                )
            else:
                return "В настоящий момент данные по курсам валют недоступны."
        except Exception as e:
            logger.error(f"General error: {e}")
            return "Не удалось получить актуальные котировки."

