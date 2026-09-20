"""
OllamaClient — thin async client for a local Ollama server.

This is the single place in the codebase that talks to Ollama. There
used to be a second, independent implementation of the same HTTP call
inside src/infrastructure/rag/generation/generator.py (duplicated
aiohttp code, its own timeout/error handling) - consolidated into this
one client so there is exactly one thing to configure, time out, and
retry.
"""
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Optional[str]:
        """Returns the generated text, or None if Ollama is unreachable/errors."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.3),
                "top_p": kwargs.get("top_p", 0.9),
                # Small models on a small VDS: keep context and output
                # short so a single request stays fast and light.
                "num_ctx": kwargs.get("num_ctx", 2048),
                "num_predict": kwargs.get("num_predict", 200),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("Ollama API error %s: %s", response.status, error_text)
                        return None
                    data = await response.json()
                    return (data.get("response") or "").strip()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning("Ollama unreachable/timeout: %s", exc)
            return None
        except Exception as exc:
            logger.error("Ollama generation error: %s", exc, exc_info=True)
            return None
