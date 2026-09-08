import aiohttp
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://172.17.0.1:11434", model: str = "tinyllama:latest"):
        self.base_url = base_url
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=120)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Генерация текста через Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_ctx": kwargs.get("num_ctx", 4096)
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return f"Ошибка Ollama: {response.status}"
        except aiohttp.ClientTimeout:
            logger.error("Ollama timeout")
            return "Превышено время ожидания ответа от Ollama."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Ошибка сервиса Ollama: {str(e)}"

    async def generate_with_context(self, prompt: str, context: str, **kwargs) -> str:
        """Генерация с контекстом."""
        full_prompt = f"{context}\n\nВопрос: {prompt}\n\nОтвет:"
        return await self.generate(full_prompt, **kwargs)
