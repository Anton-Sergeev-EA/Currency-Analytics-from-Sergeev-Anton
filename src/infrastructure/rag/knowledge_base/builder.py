import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime

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
            current_usd = df['usd_rate'].iloc[-1]
            current_eur = df['eur_rate'].iloc[-1]
            last_date = df['date'].iloc[-1]
            
            self.documents.append({
                'text': f"Текущий курс USD/RUB: {current_usd:.2f} рублей. Курс EUR/RUB: {current_eur:.2f} рублей. Данные на {last_date.strftime('%d.%m.%Y')}.",
                'source': 'cbr_current'
            })
            
            usd_mean = df['usd_rate'].mean()
            usd_std = df['usd_rate'].std()
            usd_min = df['usd_rate'].min()
            usd_max = df['usd_rate'].max()
            
            self.documents.append({
                'text': f"Статистика USD за период: среднее {usd_mean:.2f} руб., отклонение {usd_std:.2f} руб., мин {usd_min:.2f} руб., макс {usd_max:.2f} руб.",
                'source': 'stats_usd'
            })
            
            eur_mean = df['eur_rate'].mean()
            eur_std = df['eur_rate'].std()
            eur_min = df['eur_rate'].min()
            eur_max = df['eur_rate'].max()
            
            self.documents.append({
                'text': f"Статистика EUR за период: среднее {eur_mean:.2f} руб., отклонение {eur_std:.2f} руб., мин {eur_min:.2f} руб., макс {eur_max:.2f} руб.",
                'source': 'stats_eur'
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
                'text': "USD и EUR - основные резервные валюты, менее волатильны чем RUB.",
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
                'text': "Рекомендуемое соотношение: 50% RUB, 30% USD, 20% EUR для сбалансированного портфеля.",
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
