import re
from typing import Any, Dict, List, Optional

from src.application.services.forecast_service import ForecastService
from src.common.logger.logger import get_logger
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.rag.generation.generator import ResponseGenerator

logger = get_logger(__name__)

_CURRENCY_LABEL = {"usd_rate": "USD/RUB", "eur_rate": "EUR/RUB"}

# Фразы, которые выдают, что модель не ответила по существу, а вернула
# обрывок собственной инструкции/промпта (наблюдалось на слабых моделях
# на маломощном сервере - см. коммит с диагнозом искажённых ответов RAG).
_META_LEAK_MARKERS = (
    "предложение:",
    "как подумать",
    "подумайте о том",
    "краткий ответ:",
    "вопрос:",
    "данные:",
)

_MIN_CYRILLIC_RATIO = 0.6


class RAGService:
    """Сервис обработки вопросов с привлечением RAG и Ollama.

    Ответ модели никогда не возвращается пользователю "как есть": он
    проверяется в _is_usable() и, если похож на галлюцинацию, обрыв
    промпта, ответ не про ту валюту, выдуманные цифры или вообще не на
    русском языке, заменяется на детерминированный ответ, посчитанный
    напрямую из реальных данных/прогноза. Так неустойчивая маленькая
    LLM-модель не может показать пользователю в чате неправильный курс,
    валюту или ответ не на том языке.
    """

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
        analysis_keywords = [
            "курс", "доллар", "евро", "usd", "eur", "прогноз",
            "сравни", "купить", "продать", "рубл", "динамик",
        ]

        is_pure_greeting = any(g in q_lower for g in greetings) and not any(
            k in q_lower for k in analysis_keywords
        )

        if is_pure_greeting:
            return {
                "answer": (
                    "👋 Привет! Я Антон — ваш персональный финансовый аналитик.\n\n"
                    "Задайте любой вопрос по курсам валют (USD, EUR), прогнозам или их сравнению!"
                ),
                "type": "greeting",
            }

        asked_currencies = self._detect_currencies(q_lower)

        try:
            facts = await self._collect_facts()
            deterministic_answer = self._build_deterministic_answer(asked_currencies, facts)
            context = self._build_context(facts)

            llm_answer: Optional[str] = None
            try:
                llm_answer = await self.generator.generate_response(question, context)
            except Exception as e:
                logger.warning(f"Ollama generation failed, falling back to computed answer: {e}")

            if self._is_usable(llm_answer, asked_currencies, facts):
                return {"answer": llm_answer.strip(), "type": "ollama"}

            return {"answer": deterministic_answer, "type": "forecast"}

        except Exception as e:
            logger.error(f"Error in RAGService processing question: {e}", exc_info=True)
            return {
                "answer": f"Произошла ошибка при анализе данных: {str(e)}",
                "type": "error",
            }

    @staticmethod
    def _detect_currencies(q_lower: str) -> List[str]:
        """Определяет, о какой валюте именно спросили. Раньше промпт и
        разбор ответа всегда были жёстко привязаны к USD/RUB, поэтому
        вопрос про евро отвечался курсом доллара."""
        wants_usd = any(k in q_lower for k in ("доллар", "usd", "бакс"))
        wants_eur = any(k in q_lower for k in ("евро", "eur"))

        if wants_usd and not wants_eur:
            return ["usd_rate"]
        if wants_eur and not wants_usd:
            return ["eur_rate"]
        return ["usd_rate", "eur_rate"]

    async def _collect_facts(self) -> Dict[str, Dict[str, Any]]:
        df = await self.data_loader.load_data()

        facts: Dict[str, Dict[str, Any]] = {}
        for curr in ("usd_rate", "eur_rate"):
            current = None
            if df is not None and not df.empty and curr in df.columns:
                current = round(float(df[curr].iloc[-1]), 2)

            forecast = await self.forecast_service.get_forecast(days=7, currency=curr)
            forecast_list = forecast if isinstance(forecast, list) else []

            day1 = forecast_list[0]["forecast"] if forecast_list else None
            day7 = forecast_list[-1]["forecast"] if forecast_list else None

            facts[curr] = {
                "label": _CURRENCY_LABEL[curr],
                "current": current,
                "day1": day1,
                "day7": day7,
            }
        return facts

    @staticmethod
    def _build_context(facts: Dict[str, Dict[str, Any]]) -> str:
        lines = ["=== ТЕКУЩИЕ КУРСЫ И ПРОГНОЗ ЦБ РФ НА 7 ДНЕЙ ==="]
        for curr in ("usd_rate", "eur_rate"):
            f = facts.get(curr)
            if not f:
                continue
            current = f["current"] if f["current"] is not None else "неизвестно"
            day1 = f["day1"] if f["day1"] is not None else "неизвестно"
            day7 = f["day7"] if f["day7"] is not None else "неизвестно"
            lines.append(
                f"{f['label']}: текущий курс {current} руб.; "
                f"прогноз через 1 день {day1} руб.; прогноз через 7 дней {day7} руб."
            )
        return "\n".join(lines)

    @staticmethod
    def _build_deterministic_answer(
        asked_currencies: List[str], facts: Dict[str, Dict[str, Any]]
    ) -> str:
        lines = []
        for curr in asked_currencies:
            f = facts.get(curr)
            if not f or f["day1"] is None or f["day7"] is None:
                label = f["label"] if f else _CURRENCY_LABEL[curr]
                lines.append(f"Прогноз {label} сейчас недоступен.")
                continue

            lo, hi = sorted((f["day1"], f["day7"]))
            current_part = f" Текущий курс: {f['current']} руб." if f["current"] is not None else ""
            lines.append(
                f"Прогноз {f['label']} на неделю: от {lo} до {hi} руб.{current_part}"
            )
        return "\n".join(lines)

    @staticmethod
    def _is_usable(
        answer: Optional[str],
        asked_currencies: List[str],
        facts: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Отбрасывает ответы модели, которые похожи на обрыв промпта,
        повтор одной фразы, ответ не про ту валюту, которую спросили,
        называют курс, никак не похожий на реальные цифры, или выданы
        не на русском языке."""
        if not isinstance(answer, str):
            return False

        text = answer.strip()
        if len(text) < 10 or len(text) > 600:
            return False

        text_lower = text.lower()
        if any(marker in text_lower for marker in _META_LEAK_MARKERS):
            return False

        # Явное зацикливание / повтор одной и той же фразы.
        words = text_lower.split()
        if len(words) > 6 and len(set(words)) / len(words) < 0.4:
            return False

        if len(asked_currencies) == 1:
            curr = asked_currencies[0]
            other = "eur_rate" if curr == "usd_rate" else "usd_rate"
            target_mentioned = RAGService._mentions_currency(text_lower, curr)
            other_mentioned = RAGService._mentions_currency(text_lower, other)
            # Спросили конкретно про одну валюту - ответ не должен ни
            # молчать про неё, ни притягивать данные другой валюты
            # (модель наблюдалась смешивающей обе в одном ответе).
            if not target_mentioned or other_mentioned:
                return False

        if not RAGService._numbers_are_plausible(text, asked_currencies, facts):
            return False

        if not RAGService._is_mostly_russian(text):
            return False

        return True

    @staticmethod
    def _mentions_currency(text_lower: str, curr: str) -> bool:
        if curr == "eur_rate":
            return bool(re.search(r"евро|eur", text_lower))
        return bool(re.search(r"доллар|usd", text_lower))

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Числа из текста ответа, с поддержкой и точки, и запятой как
        десятичного разделителя (модель использует оба варианта)."""
        numbers = []
        for raw in re.findall(r"\d+[.,]\d+|\d+", text):
            try:
                numbers.append(float(raw.replace(",", ".")))
            except ValueError:
                continue
        return numbers

    @staticmethod
    def _numbers_are_plausible(
        text: str, asked_currencies: List[str], facts: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Проверяет, что курсы, которые называет модель, похожи на
        реальные (в пределах 20% от известных нам значений). Не блокирует
        ответ, если в нём вообще нет чисел похожих на курс валюты -
        такие тексты просто ничего не утверждают о конкретной цифре."""
        reference_values = [
            float(f[key])
            for curr in asked_currencies
            for f in (facts.get(curr),)
            if f
            for key in ("current", "day1", "day7")
            if f.get(key) is not None
        ]
        if not reference_values:
            return True

        # Малые числа ("1 день", "7 дней", номера пунктов) не являются
        # курсом валюты и не должны участвовать в проверке.
        mentioned = [n for n in RAGService._extract_numbers(text) if n > 10]
        if not mentioned:
            return True

        tolerance = 0.2
        return any(
            abs(n - ref) / ref <= tolerance
            for n in mentioned
            for ref in reference_values
        )

    @staticmethod
    def _is_mostly_russian(text: str) -> bool:
        """tinyllama иногда полностью игнорирует инструкцию отвечать на
        русском и отвечает на английском - такой ответ пользователю не
        подходит, даже если по цифрам он верный."""
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return True
        cyrillic = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
        return (cyrillic / len(letters)) >= _MIN_CYRILLIC_RATIO
