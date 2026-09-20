"""
RAGService — question answering grounded in real retrieval and real numbers.

Previous behaviour worth calling out (this file's own history): it never
actually retrieved anything, despite a whole vector-store/knowledge-base
stack existing elsewhere in the codebase (src/infrastructure/rag/) - it
just stuffed the latest rates and a forecast into a prompt and asked a
tiny local LLM (`tinyllama` by default) to both do the reasoning and the
arithmetic. Small local models are not reliable at either.

This version splits the work by what each part is actually good at:
  - Numbers (currency conversion, investment projections, forecast
    comparisons) are computed deterministically in FinanceAdvisor from
    the real current rate and the real ML forecast - never guessed by
    an LLM.
  - Open-ended questions are answered by retrieving the most relevant
    documents from the knowledge base (VectorStore, TF-IDF-based) and
    asking Ollama to phrase an answer grounded in *only* that retrieved
    text - genuine retrieval-augmented generation, with the retrieved
    sources returned alongside the answer so a caller (or the frontend)
    can show what it was actually grounded in.
  - If Ollama is unreachable, degrades to a plain templated answer built
    from the retrieved documents rather than failing outright.
"""
from datetime import datetime
from typing import Any, Dict

from src.application.services.data_service import DataService
from src.application.services.finance_advisor import FinanceAdvisor
from src.application.services.forecast_service import ForecastService
from src.core.config import settings
from src.core.constants import SUPPORTED_CURRENCIES, short_code
from src.common.logger.logger import get_logger
from src.infrastructure.data.loader import DataLoader
from src.infrastructure.rag.knowledge_base.builder import KnowledgeBaseBuilder
from src.infrastructure.rag.llm.ollama_client import OllamaClient
from src.infrastructure.rag.vector_store.embeddings import VectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Ты — Антон, финансовый ассистент проекта Currency Analytics. "
    "Отвечай кратко (2-4 предложения) и по-русски, используя ТОЛЬКО факты "
    "из раздела «Контекст». Если контекста недостаточно, честно скажи об "
    "этом. Никогда не придумывай числа сам — все числа в контексте уже "
    "рассчитаны заранее. Ты не даёшь инвестиционных советов, только "
    "информацию для принятия решений."
)


class RAGService:
    """Сервис обработки вопросов с реальным retrieval и Ollama."""

    def __init__(self):
        self.forecast_service = ForecastService()
        self.data_service = DataService()
        self.data_loader = DataLoader()
        self.finance_advisor = FinanceAdvisor()
        self.vector_store = VectorStore()
        self.kb_builder = KnowledgeBaseBuilder()
        self.ollama = OllamaClient(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)
        self._kb_built_date: str | None = None

    async def ask(self, question: str) -> Dict[str, Any]:
        """Основной метод, вызываемый API роутером."""
        return await self.process_question(question)

    async def _ensure_knowledge_base(self, df) -> None:
        """(Re)indexes the knowledge base once per day, or if empty."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._kb_built_date == today and self.vector_store.count() > 0:
            return
        try:
            self.kb_builder.clear_documents()
            self.kb_builder.build_currency_knowledge(df)
            self.kb_builder.build_investment_knowledge()
            documents = self.kb_builder.get_documents()
            if documents:
                self.vector_store.index_documents(documents)
                self._kb_built_date = today
        except Exception as exc:
            logger.warning("Knowledge base build failed: %s", exc)

    async def process_question(self, question: str) -> Dict[str, Any]:
        q_lower = question.lower().strip()

        greetings = ["привет", "здравствуй", "добрый день", "добрый вечер", "кто ты"]
        analysis_keywords = ["курс", "доллар", "евро", "юан", "фунт", "usd", "eur", "cny", "gbp",
                              "прогноз", "сравни", "купить", "продать", "рубл", "динамик",
                              "влож", "заработа", "конверт", "переведи"]
        is_pure_greeting = any(g in q_lower for g in greetings) and not any(k in q_lower for k in analysis_keywords)

        if is_pure_greeting:
            return {
                "answer": (
                    "👋 Привет! Я Антон — ваш финансовый ассистент.\n\n"
                    "Спросите про курс USD/EUR, прогноз на конкретный срок, "
                    "конвертацию суммы или сравнение валют — отвечу точными "
                    "расчётами, а не догадками."
                ),
                "type": "greeting",
                "sources": [],
            }

        try:
            df = await self.data_loader.load_data()
            await self._ensure_knowledge_base(df)

            current_rates = {}
            if df is not None and not df.empty:
                for col in SUPPORTED_CURRENCIES:
                    if col in df.columns:
                        current_rates[short_code(col)] = float(df[col].iloc[-1])

            intent = self.finance_advisor.detect_intent(question)

            if intent == "conversion":
                result = self.finance_advisor.compute_conversion(question, current_rates)
                if result:
                    answer = (
                        f"{result['amount']:,.0f} {result['from']} ≈ {result['result']:,.2f} {result['to']} "
                        f"по текущему курсу ЦБ РФ {result['rate']:.2f}."
                    ).replace(",", " ")
                    return {"answer": answer, "type": "conversion", "sources": ["cbr_current"], "confidence": 0.95}

            if intent in ("investment", "comparison", "forecast"):
                forecasts = {}
                for col in SUPPORTED_CURRENCIES:
                    if short_code(col) in current_rates:
                        forecasts[short_code(col)] = await self.forecast_service.get_forecast(days=30, currency=col)

                if intent == "investment":
                    result = self.finance_advisor.compute_investment(question, current_rates, forecasts)
                    if result:
                        sign = "прибыль" if result["profit_rub"] >= 0 else "убыток"
                        answer = (
                            f"Если сегодня обменять {result['amount']:,.0f} ₽ на {result['currency']} "
                            f"по курсу {result['current_rate']:.2f}, а через {result['horizon_days']} дн. курс "
                            f"(по прогнозу ML-модели) составит ≈{result['forecast_rate']:.2f}, то итоговая сумма "
                            f"в рублях будет ≈{result['future_value_rub']:,.0f} ₽ "
                            f"({sign} {abs(result['profit_rub']):,.0f} ₽, {result['profit_pct']:+.2f}%).\n"
                            f"Это прогноз, а не гарантия — реальный курс может отличаться."
                        ).replace(",", " ")
                        return {"answer": answer, "type": "investment", "sources": ["forecast_ml"], "confidence": 0.7}

                if intent == "comparison":
                    result = self.finance_advisor.compute_comparison(current_rates, forecasts)
                    if result:
                        parts = []
                        for ccy, info in result.items():
                            parts.append(
                                f"{ccy.upper()}: {info['current_rate']:.2f} → {info['forecast_rate']:.2f} "
                                f"через {info['horizon_days']} дн. ({info['change_pct']:+.2f}%)"
                            )
                        winner = max(result.items(), key=lambda kv: kv[1]["change_pct"])[0].upper()
                        answer = "Прогноз ML-модели: " + "; ".join(parts) + f".\nБольший рост прогнозируется у {winner}."
                        return {"answer": answer, "type": "comparison", "sources": ["forecast_ml"], "confidence": 0.7}

                if intent == "forecast":
                    result = self.finance_advisor.compute_forecast(question, current_rates, forecasts)
                    if result:
                        parts = []
                        for ccy, info in result.items():
                            bound = ""
                            if info.get("lower_bound") is not None and info.get("upper_bound") is not None:
                                bound = f" (доверительный интервал {info['lower_bound']:.2f}–{info['upper_bound']:.2f})"
                            parts.append(
                                f"{ccy.upper()}: сейчас {info['current_rate']:.2f} → прогноз {info['forecast_rate']:.2f} "
                                f"через {info['horizon_days']} дн. ({info['change_pct']:+.2f}%){bound}"
                            )
                        answer = "Прогноз ансамблевой ML-модели:\n" + "\n".join(parts)
                        return {"answer": answer, "type": "forecast", "sources": ["forecast_ml"], "confidence": 0.75}

            # General / open-ended question: retrieve from the knowledge base
            # and let the LLM phrase an answer grounded in what was retrieved.
            retrieved = self.vector_store.search_with_scores(question, n_results=3)
            if not retrieved:
                return {
                    "answer": "Не нашёл релевантной информации по вашему вопросу. "
                              "Попробуйте спросить про курс, прогноз, сравнение валют или расчёт по конкретной сумме.",
                    "type": "general",
                    "sources": [],
                    "confidence": 0.3,
                }

            context = "\n".join(f"- {text}" for text, _source, _score in retrieved)
            sources = list({source for _text, source, _score in retrieved})
            prompt = f"Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"

            llm_answer = await self.ollama.generate(prompt, system_prompt=SYSTEM_PROMPT)
            if llm_answer:
                return {"answer": llm_answer, "type": "general", "sources": sources, "confidence": 0.6}

            # Ollama unreachable: still useful, degrade to the raw retrieved facts.
            return {
                "answer": "Локальная LLM (Ollama) сейчас недоступна, но вот что нашлось по вашему вопросу:\n" + context,
                "type": "general",
                "sources": sources,
                "confidence": 0.4,
            }

        except Exception as e:
            logger.error(f"Error in RAGService processing question: {e}", exc_info=True)
            return {
                "answer": f"Произошла ошибка при анализе данных: {str(e)}",
                "type": "error",
                "sources": [],
            }
