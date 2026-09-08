import openai
import logging
from typing import Optional
from src.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL
        self.model = settings.LLM_MODEL
        self.enabled = settings.LLM_ENABLED
        
        if self.enabled and self.api_key and self.api_key != "your_openai_api_key_here":
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"LLM Client initialized with model: {self.model}")
        else:
            self.client = None
            logger.warning("LLM Client disabled or no API key provided")
    
    async def generate(self, prompt: str, context: str) -> Optional[str]:
        if not self.enabled or not self.client:
            logger.warning("LLM is not enabled, returning None")
            return None
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {prompt}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        return """Ты - Антон, профессиональный AI-ассистент по валютному рынку и инвестициям.
        Отвечай на русском языке, используй предоставленный контекст.
        Если данных недостаточно - честно скажи об этом.
        Давай четкие, структурированные ответы.
        Не давай финансовых рекомендаций, только аналитику.
        Всегда указывай источник данных, если это возможно.
        Ты не даешь инвестиционных советов, а только предоставляешь информацию для принятия решений.
        """
