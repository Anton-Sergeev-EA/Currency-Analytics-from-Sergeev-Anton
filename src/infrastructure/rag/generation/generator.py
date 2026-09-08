import aiohttp
import asyncio
import logging
import re
from src.core.config import settings

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        self.model = settings.OLLAMA_MODEL

    async def generate_response(self, question: str, context: str) -> str:
        # Извлекаем только нужные данные из контекста
        usd = "неизвестно"
        eur = "неизвестно"
        forecast_start = "неизвестно"
        forecast_end = "неизвестно"
        
        # Ищем курсы
        usd_match = re.search(r'USD/RUB:\s*([\d.]+)', context)
        if usd_match:
            usd = usd_match.group(1)
        
        eur_match = re.search(r'EUR/RUB:\s*([\d.]+)', context)
        if eur_match:
            eur = eur_match.group(1)
        
        # Ищем прогноз
        forecast_match = re.search(r'ПРОГНОЗ.*?(\d+\.\d+).*?(\d+\.\d+)', context, re.DOTALL)
        if forecast_match:
            forecast_start = forecast_match.group(1)
            forecast_end = forecast_match.group(2)
        else:
            # Если прогноз не найден, ищем даты с курсами
            dates = re.findall(r'"date":\s*"([^"]+)"', context)
            rates = re.findall(r'"rate":\s*([\d.]+)', context)
            if len(rates) >= 2:
                forecast_start = rates[0]
                forecast_end = rates[-1]
            elif usd != "неизвестно":
                forecast_start = usd
                forecast_end = usd
        
        # Формируем чистый промпт
        prompt = f"""Ты финансовый аналитик. Отвечай кратко (2-3 предложения) на русском языке.

Данные:
- Текущий курс USD/RUB: {usd} руб.
- Текущий курс EUR/RUB: {eur} руб.
- Прогноз USD/RUB на неделю: от {forecast_start} до {forecast_end} руб.

Вопрос: {question}

Краткий ответ:"""

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 80}
            }
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.ollama_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = data.get("response", "")
                        # Очищаем ответ
                        answer = answer.replace("Краткий ответ:", "").strip()
                        answer = answer.replace("Ответ:", "").strip()
                        answer = answer.replace("ОТВЕТ:", "").strip()
                        answer = answer.replace("Данные:", "").strip()
                        answer = answer.replace("Вопрос:", "").strip()
                        # Обрезаем до 2-3 предложений
                        sentences = answer.split('.')
                        if len(sentences) > 3:
                            answer = '.'.join(sentences[:3]) + '.'
                        # Если ответ слишком короткий или пустой
                        if len(answer) < 10:
                            return f"Прогноз USD/RUB на неделю: {forecast_start} - {forecast_end} руб."
                        return answer
                    return "Ошибка Ollama"
        except asyncio.TimeoutError:
            return "Превышено время ожидания"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Ошибка: {str(e)}"

