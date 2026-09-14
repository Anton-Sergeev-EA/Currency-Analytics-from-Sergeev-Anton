import logging
import re

import aiohttp

from src.core.config import settings

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Тонкая обёртка над Ollama.

    Раньше этот класс сам вырезал регэксом курсы/прогноз из общего
    текстового контекста и всегда подставлял их в жёстко прописанную
    строку "Прогноз USD/RUB на неделю: ..." - из-за этого ответ про
    евро подписывался как курс доллара, а сам регэксп цеплял первые
    попавшиеся числа из контекста. Теперь готовый контекст с реальными
    цифрами по обеим валютам собирает RAGService, а здесь только вызов
    модели и минимальная чистка текста. RAGService сам решает, можно ли
    доверять ответу модели (см. _is_usable), и заменяет его посчитанным
    напрямую из данных, если ответ похож на галлюцинацию или обрыв
    промпта - такое наблюдалось на маленьких моделях на слабом сервере.
    """

    def __init__(self):
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        self.model = settings.OLLAMA_MODEL

    async def generate_response(self, question: str, context: str) -> str:
        prompt = f"""Ты финансовый аналитик Антон. Отвечай кратко (2-3 предложения), на русском языке, строго по вопросу и только на основе приведённых данных. Не придумывай цифры и не путай валюты между собой.

{context}

Вопрос: {question}

Краткий ответ:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 120},
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.ollama_url, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama вернул статус {resp.status}")
                data = await resp.json()

        answer = data.get("response", "")
        for marker in ("Краткий ответ:", "Ответ:", "ОТВЕТ:", "Данные:", "Вопрос:"):
            answer = answer.replace(marker, "")
        answer = answer.strip()

        # Режем на предложения по точке, которая НЕ является десятичным
        # разделителем (не за ней сразу цифра) - иначе "0.24 руб." рвётся
        # прямо посередине числа и ответ обрывается на полуслове.
        sentences = re.split(r"\.(?!\d)", answer)
        if len(sentences) > 3:
            answer = ".".join(s for s in sentences[:3] if s.strip()) + "."

        return answer.strip()
