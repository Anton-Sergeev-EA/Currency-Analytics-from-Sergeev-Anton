import logging
import pandas as pd
from typing import List, Dict

from src.core.constants import SUPPORTED_CURRENCIES, short_code

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    def __init__(self):
        self.documents = []

    def build_currency_knowledge(self, df: pd.DataFrame) -> List[Dict]:
        self.documents = []

        if df is None or len(df) == 0:
            logger.warning("No data available for knowledge base")
            return self.documents

        try:
            present = [c for c in SUPPORTED_CURRENCIES if c in df.columns]
            last_date = df['date'].iloc[-1]

            current_line = ", ".join(
                f"{short_code(c).upper()}/RUB: {df[c].iloc[-1]:.2f} рублей" for c in present
            )
            self.documents.append({
                'text': f"Текущие курсы ({current_line}). Данные на {last_date.strftime('%d.%m.%Y')}.",
                'source': 'cbr_current',
            })

            for col in present:
                code = short_code(col).upper()
                mean, std = df[col].mean(), df[col].std()
                lo, hi = df[col].min(), df[col].max()
                self.documents.append({
                    'text': f"Статистика {code} за период: среднее {mean:.2f} руб., "
                            f"отклонение {std:.2f} руб., мин {lo:.2f} руб., макс {hi:.2f} руб.",
                    'source': f'stats_{code.lower()}',
                })

            logger.info(f"Built {len(self.documents)} currency knowledge documents")

        except Exception as e:
            logger.error(f"Error building currency knowledge: {e}")

        return self.documents

    def build_investment_knowledge(self) -> List[Dict]:
        investment_tips = [
            {
                'text': "Диверсификация портфеля: распределяйте инвестиции между разными валютами для снижения рисков.",
                'source': 'expert_investment'
            },
            {
                'text': "Долгосрочные инвестиции в валюту имеют меньшую волатильность, чем краткосрочные спекуляции.",
                'source': 'expert_investment'
            },
            {
                'text': "Ключевые факторы влияющие на курс рубля: цена нефти, инфляция, ставка ЦБ, геополитика.",
                'source': 'expert_economic'
            },
            {
                'text': "Метод усреднения стоимости (DCA): покупка валюты регулярными суммами снижает риск.",
                'source': 'expert_strategy'
            },
            {
                'text': "USD, EUR и GBP - основные резервные валюты, юань (CNY) набирает вес во внешней торговле России.",
                'source': 'expert_currency'
            },
            {
                'text': "При покупке валюты обращайте внимание на спред (разница цены покупки и продажи).",
                'source': 'expert_practical'
            },
            {
                'text': "Следите за макроэкономическими показателями: ВВП, безработица, инфляция.",
                'source': 'expert_economic'
            },
            {
                'text': "Избегайте эмоциональных решений, придерживайтесь стратегии.",
                'source': 'expert_psychology'
            },
            {
                'text': "Рекомендуемое соотношение для диверсифицированного портфеля: часть в рублях, часть распределена между USD, EUR, CNY и GBP.",
                'source': 'expert_portfolio'
            }
        ]

        for tip in investment_tips:
            self.documents.append(tip)

        logger.info(f"Built {len(investment_tips)} investment knowledge documents")
        return self.documents

    def get_documents(self) -> List[Dict]:
        return self.documents

    def clear_documents(self):
        self.documents = []
