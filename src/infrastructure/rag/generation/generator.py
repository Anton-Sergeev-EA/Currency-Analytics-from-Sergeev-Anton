import re
import logging

logger = logging.getLogger(__name__)


class RAGGenerator:
    def __init__(self):
        from src.application.services.data_service import DataService
        from src.application.services.forecast_service import ForecastService
        self.data_service = DataService()
        self.forecast_service = ForecastService()
        logger.info("RAGGenerator initialized")

    def generate_response(self, question: str) -> dict:
        logger.info(f"Generating response for: {question[:50]}...")

        q = question.lower().strip()

        has_forecast = any(word in q for word in ['прогноз', 'будет', 'через', 'ожидать', 'перспектив', 'предсказание'])
        has_profit = any(word in q for word in ['прибыль', 'доход', 'заработаю', 'вложу', 'инвестиция', 'сколько'])
        has_comparison = any(word in q for word in ['сравни', 'лучше', 'отличие', 'какая валюта', 'что выбрать'])

        try:
            if has_forecast:
                return self._handle_forecast(question)
            elif has_profit:
                return self._handle_profit(question)
            elif has_comparison:
                return self._handle_comparison()
            else:
                return self._handle_general(question)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {
                'answer': f'Ошибка: {str(e)}',
                'type': 'error',
                'confidence': 0.0
            }

    def _handle_forecast(self, question: str) -> dict:
        try:
            logger.info(f"Forecast question: {question}")

            q = question.lower()
            is_usd = 'доллар' in q or 'usd' in q or 'бакс' in q
            is_eur = 'евро' in q or 'eur' in q or 'euro' in q

            logger.info(f"is_usd={is_usd}, is_eur={is_eur}")

            forecast_data = self.forecast_service.get_forecast_with_uncertainty(30)

            if is_usd and not is_eur:
                d = forecast_data['usd_rate']
                answer = "Прогноз USD/RUB на 30 дней:\n"
                answer += f"Текущий:  {d['mean'][0]:.2f} RUB\n"
                answer += f"Через 7 дней:  {d['mean'][7]:.2f} RUB\n"
                answer += f"Через 14 дней: {d['mean'][14]:.2f} RUB\n"
                answer += f"Через 30 дней: {d['mean'][-1]:.2f} RUB\n"
                change = (d['mean'][-1] - d['mean'][0]) / d['mean'][0] * 100
                direction = "укреплению" if change > 0 else "ослаблению"
                answer += f"\nОжидается {direction} на {abs(change):.1f}%"
                answer += f"\nДиапазон: {d['lower_bound'][-1]:.2f} - {d['upper_bound'][-1]:.2f} RUB"
                return {'answer': answer, 'type': 'forecast', 'confidence': 0.85}

            if is_eur and not is_usd:
                d = forecast_data['eur_rate']
                answer = "Прогноз EUR/RUB на 30 дней:\n"
                answer += f"Текущий:  {d['mean'][0]:.2f} RUB\n"
                answer += f"Через 7 дней:  {d['mean'][7]:.2f} RUB\n"
                answer += f"Через 14 дней: {d['mean'][14]:.2f} RUB\n"
                answer += f"Через 30 дней: {d['mean'][-1]:.2f} RUB\n"
                change = (d['mean'][-1] - d['mean'][0]) / d['mean'][0] * 100
                direction = "укреплению" if change > 0 else "ослаблению"
                answer += f"\nОжидается {direction} на {abs(change):.1f}%"
                answer += f"\nДиапазон: {d['lower_bound'][-1]:.2f} - {d['upper_bound'][-1]:.2f} RUB"
                return {'answer': answer, 'type': 'forecast', 'confidence': 0.85}

            usd = forecast_data['usd_rate']
            eur = forecast_data['eur_rate']
            answer = "Прогноз курсов на 30 дней:\n"
            answer += "USD/RUB:\n"
            answer += f"Текущий:  {usd['mean'][0]:.2f} RUB\n"
            answer += f"Через 30: {usd['mean'][-1]:.2f} RUB\n"
            answer += f"Изменение: {((usd['mean'][-1] - usd['mean'][0]) / usd['mean'][0] * 100):+.1f}%\n\n"
            answer += "EUR/RUB:\n"
            answer += f"Текущий:  {eur['mean'][0]:.2f} RUB\n"
            answer += f"Через 30: {eur['mean'][-1]:.2f} RUB\n"
            answer += f"Изменение: {((eur['mean'][-1] - eur['mean'][0]) / eur['mean'][0] * 100):+.1f}%"
            return {'answer': answer, 'type': 'forecast', 'confidence': 0.85}

        except Exception as e:
            logger.error(f"Forecast error: {e}", exc_info=True)
            return self._handle_general(question)

    def _handle_profit(self, question: str) -> dict:
        try:
            amount = 1000
            numbers = re.findall(r'\d+[,.]?\d*', question.replace(',', '.'))
            if numbers:
                amount = float(numbers[0])

            rates = self.data_service.get_current_rates()
            forecast_data = self.forecast_service.get_forecast_with_uncertainty(30)

            usd_profit = (forecast_data['usd_rate']['mean'][-1] - rates['usd']) / rates['usd'] * 100
            eur_profit = (forecast_data['eur_rate']['mean'][-1] - rates['eur']) / rates['eur'] * 100

            answer = f"Инвестиция {amount:,.0f} RUB:\n"
            answer += f"USD: доходность {usd_profit:+.1f}%, прибыль {((forecast_data['usd_rate']['mean'][-1] - rates['usd']) * amount):+,.0f} RUB\n"
            answer += f"EUR: доходность {eur_profit:+.1f}%, прибыль {((forecast_data['eur_rate']['mean'][-1] - rates['eur']) * amount):+,.0f} RUB"
            return {'answer': answer, 'type': 'profit', 'confidence': 0.8}

        except Exception as e:
            logger.error(f"Profit error: {e}", exc_info=True)
            return self._handle_general(question)

    def _handle_comparison(self) -> dict:
        try:
            rates = self.data_service.get_current_rates()
            df = self.data_service.get_historical_data(90)

            usd_vol = df['usd_rate'].std()
            eur_vol = df['eur_rate'].std()

            answer = "Сравнение валют:\n"
            answer += f"USD: {rates['usd']:.2f} RUB, волатильность {usd_vol:.4f}\n"
            answer += f"EUR: {rates['eur']:.2f} RUB, волатильность {eur_vol:.4f}\n"

            if usd_vol > eur_vol:
                answer += "\nUSD более волатилен (выше риск)"
            else:
                answer += "\nEUR более волатилен (выше риск)"

            return {'answer': answer, 'type': 'comparison', 'confidence': 0.9}

        except Exception as e:
            logger.error(f"Comparison error: {e}", exc_info=True)
            return self._handle_general("")

    def _handle_general(self, question: str = "") -> dict:
        try:
            rates = self.data_service.get_current_rates()

            answer = "Анализ валют:\n"
            answer += f"USD/RUB: {rates['usd']:.2f} RUB\n"
            answer += f"EUR/RUB: {rates['eur']:.2f} RUB\n\n"
            answer += "Задайте вопрос:\n"
            answer += "'Какой прогноз по доллару?'\n"
            answer += "'Какой прогноз по евро?'\n"
            answer += "'Сколько я заработаю на 1000 рублей?'\n"
            answer += "'Сравни доллар и евро'"

            return {
                'answer': answer,
                'type': 'general',
                'confidence': 0.5
            }

        except Exception as e:
            logger.error(f"General error: {e}", exc_info=True)
            return {
                'answer': f"Ошибка: {str(e)}",
                'type': 'error',
                'confidence': 0.0
            }
