import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# `from ..config import settings` used to be here, importing a module that
# no longer exists (src/config.py was consolidated into src/core/config.py
# - see that file's docstring) and that this class never actually used.
# It broke `import src.ab_testing.ab_service` outright, which meant
# scripts/run_ab_production.py and scripts/update_ab_actual_rates.py
# (both `from ab_testing.ab_service import get_ab_service`) couldn't run.
from ..infrastructure.ml.models.ensemble import EnsembleModel

Base = declarative_base()

class ABTestLog(Base):
    """Таблица для логирования A/B-тестов."""
    __tablename__ = 'ab_test_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    user_id = Column(String(255))
    session_id = Column(String(255))
    variant = Column(String(10))  # 'A' или 'B'.
    model_type = Column(String(50))  # 'ensemble_v1' или 'ensemble_v2'.
    predicted_rate = Column(Float)
    actual_rate = Column(Float, nullable=True)  # Заполняется позже, когда курс станет известен.
    forecast_date = Column(DateTime)  # На какую дату был прогноз.
    is_converted = Column(Boolean, default=False)  # Для UI-метрик.

class ABTestService:
    """
    Сервис для управления A/B-тестированием в продакшене.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        # Используем SQLite для простоты, можно заменить на вашу БД.
        if db_url is None:
            db_path = Path("data/ab_testing.db")
            db_path.parent.mkdir(exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Создаем экземпляры моделей
        self.model_a = EnsembleModel()
        self.model_b = EnsembleModel()
        
        # Загружаем модели, если файлы существуют
        model_v1_path = Path('models/ensemble_v1.pkl')
        model_v2_path = Path('models/ensemble_v2.pkl')
        
        if model_v1_path.exists():
            self.model_a.load(str(model_v1_path))
        else:
            print(f"Warning: Model file {model_v1_path} not found. Using untrained model.")
            
        if model_v2_path.exists():
            self.model_b.load(str(model_v2_path))
        else:
            print(f"Warning: Model file {model_v2_path} not found. Using untrained model.")
        
        # Конфигурация теста.
        self.traffic_split = 0.5  # 50% на каждую модель.
        self.variant_mapping = {
            'A': {'model': self.model_a, 'name': 'ensemble_v1'},
            'B': {'model': self.model_b, 'name': 'ensemble_v2'}
        }
        
    def get_variant_for_user(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Определяет, какую модель показывать пользователю.
        Использует детерминированное распределение для консистентности.
        """
        # Детерминированное назначение на основе user_id (если есть)
        if user_id:
            hash_val = int(hashlib.md5(f"{user_id}_ab_test".encode()).hexdigest(), 16)
            variant = 'A' if (hash_val % 100) < (self.traffic_split * 100) else 'B'
        # Или на основе session_id.
        elif session_id:
            hash_val = int(hashlib.md5(f"{session_id}_ab_test".encode()).hexdigest(), 16)
            variant = 'A' if (hash_val % 100) < (self.traffic_split * 100) else 'B'
        else:
            # Случайное назначение, если ничего нет.
            variant = random.choices(['A', 'B'], weights=[self.traffic_split, 1-self.traffic_split])[0]
        
        # Логируем назначение.
        self._log_assignment(variant, user_id, session_id)
        
        return {
            'variant': variant,
            'model_name': self.variant_mapping[variant]['name'],
            'model': self.variant_mapping[variant]['model']
        }
    
    def _log_assignment(self, variant: str, user_id: str, session_id: str):
        """Логирует назначение пользователя в группу."""
        session = self.Session()
        try:
            log_entry = ABTestLog(
                user_id=user_id or 'anonymous',
                session_id=session_id or 'unknown',
                variant=variant,
                model_type=self.variant_mapping[variant]['name'],
                is_converted=False
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging AB assignment: {e}")
        finally:
            session.close()
    
    def log_prediction(self, user_id: str, session_id: str, variant: str, 
                       predicted_rate: float, forecast_date: datetime):
        """Обновляет запись с предсказанием."""
        session = self.Session()
        try:
            # Находим последнюю запись для этого пользователя.
            log_entry = session.query(ABTestLog).filter(
                ABTestLog.user_id == (user_id or 'anonymous'),
                ABTestLog.session_id == (session_id or 'unknown'),
                ABTestLog.variant == variant
            ).order_by(ABTestLog.timestamp.desc()).first()
            
            if log_entry:
                log_entry.predicted_rate = predicted_rate
                log_entry.forecast_date = forecast_date
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging prediction: {e}")
        finally:
            session.close()
    
    def update_actual_rates(self):
        """Обновляет фактические курсы для завершенных прогнозов."""
        session = self.Session()
        try:
            # Заглушка: логика получения фактических курсов не реализована.
            # В реальности здесь нужно запросить прогнозы без известного
            # фактического курса и обновить их, например:
            #
            #   pending = session.query(ABTestLog).filter(
            #       ABTestLog.forecast_date.isnot(None),
            #       ABTestLog.actual_rate.is_(None)
            #   ).all()
            #   for log in pending:
            #       log.actual_rate = get_actual_rate(log.forecast_date)
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error updating actual rates: {e}")
        finally:
            session.close()
    
    def get_ab_stats(self, days: int = 30) -> Dict[str, Any]:
        """Получает статистику A/B-теста за последние N дней"""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Получаем все логи за период.
            logs = session.query(ABTestLog).filter(
                ABTestLog.timestamp >= cutoff_date,
                ABTestLog.actual_rate.isnot(None),
                ABTestLog.predicted_rate.isnot(None)
            ).all()
            
            df = pd.DataFrame([{
                'variant': log.variant,
                'predicted': log.predicted_rate,
                'actual': log.actual_rate,
                'model_type': log.model_type
            } for log in logs])
            
            if df.empty:
                return {'error': 'No data available for analysis'}
            
            # Рассчитываем метрики по группам.
            stats = {}
            for variant in ['A', 'B']:
                group = df[df['variant'] == variant]
                if not group.empty:
                    mae = np.mean(np.abs(group['predicted'] - group['actual']))
                    mape = np.mean(np.abs((group['predicted'] - group['actual']) / group['actual'])) * 100
                    stats[f'variant_{variant}'] = {
                        'count': len(group),
                        'mae': mae,
                        'mape': mape,
                        'mean_predicted': group['predicted'].mean(),
                        'mean_actual': group['actual'].mean()
                    }
            
            # Статистическая значимость (t-test).
            if 'variant_A' in stats and 'variant_B' in stats:
                errors_a = df[df['variant'] == 'A']['predicted'] - df[df['variant'] == 'A']['actual']
                errors_b = df[df['variant'] == 'B']['predicted'] - df[df['variant'] == 'B']['actual']
                
                # scipy is a listed dependency (requirements.txt) specifically
                # for this ttest_ind call - it wasn't before, so this branch
                # (both variants having data) raised ModuleNotFoundError.
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_ind(errors_a, errors_b)
                stats['statistical_significance'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            session.close()

_ab_service = None

def get_ab_service() -> ABTestService:
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestService()
    return _ab_service
