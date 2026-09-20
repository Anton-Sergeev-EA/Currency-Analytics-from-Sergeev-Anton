"""
ABTestService — real A/B testing between the ML ensemble and a naive baseline.

## What changed and why

The presentation layer (src/presentation/ab_testing/routes.py) used to
return fixed mock JSON for every endpoint — a fake "1000 requests, +0.69%
improvement" no matter what happened. This service was written *earlier*
and was already real (deterministic hash-based assignment, a real SQLite
log, a real t-test) — it just was never wired up to those routes at all.
It is now.

The other change is what the two variants actually are. The original
design loaded two full independent `EnsembleModel` instances
(`ensemble_v1.pkl` / `ensemble_v2.pkl`) — two copies of 4 tree-based
regressors each, resident in memory the whole time, for files that don't
exist yet. On a 4GB-RAM VDS that's a wasteful way to compare two
essentially-identical, both-untrained models.

Variant A is instead the project's one real trained ensemble (loaded
on demand via ForecastService — the same model the rest of the app uses,
so there's no second copy sitting in memory). Variant B is a persistence
/ random-walk baseline: "tomorrow's rate = today's rate". This is the
standard benchmark in FX forecasting (exchange rates are close enough to
a random walk that beating "no change" is a genuinely meaningful test),
it costs nothing to compute, and the comparison it produces — does the ML
ensemble actually add value over the naive baseline? — is a real,
useful, portfolio-relevant question. See ForecastService.get_forecast()
and .get_persistence_baseline().
"""
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

from src.application.services.forecast_service import ForecastService

Base = declarative_base()


class ABTestLog(Base):
    """Таблица для логирования A/B-тестов."""
    __tablename__ = 'ab_test_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    user_id = Column(String(255))
    session_id = Column(String(255))
    variant = Column(String(10))  # 'A' (ML ensemble) или 'B' (persistence baseline).
    currency = Column(String(20), default="usd_rate")
    model_type = Column(String(50))  # 'ml_ensemble' или 'persistence_baseline'.
    predicted_rate = Column(Float)
    actual_rate = Column(Float, nullable=True)  # Заполняется позже, когда курс станет известен.
    forecast_date = Column(DateTime)  # На какую дату был прогноз.
    is_converted = Column(Boolean, default=False)  # Для UI-метрик.


VARIANT_MODEL_NAMES = {"A": "ml_ensemble", "B": "persistence_baseline"}


class ABTestService:
    """Сервис для управления A/B-тестированием модели прогноза в продакшене."""

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_path = Path("data/ab_testing.db")
            db_path.parent.mkdir(exist_ok=True)
            db_url = f"sqlite:///{db_path}"

        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Reused, not duplicated: ForecastService loads the trained model
        # from disk lazily per call rather than keeping a second
        # long-lived copy in memory (see module docstring).
        self.forecast_service = ForecastService()

        self.traffic_split = 0.5  # доля трафика на вариант A.

    def get_variant_for_user(self, user_id: Optional[str], session_id: Optional[str]) -> Dict[str, Any]:
        """
        Определяет, какую модель показывать пользователю.
        Использует детерминированное распределение для консистентности.
        """
        if user_id:
            hash_val = int(hashlib.md5(f"{user_id}_ab_test".encode()).hexdigest(), 16)
            variant = 'A' if (hash_val % 100) < (self.traffic_split * 100) else 'B'
        elif session_id:
            hash_val = int(hashlib.md5(f"{session_id}_ab_test".encode()).hexdigest(), 16)
            variant = 'A' if (hash_val % 100) < (self.traffic_split * 100) else 'B'
        else:
            variant = random.choices(['A', 'B'], weights=[self.traffic_split, 1 - self.traffic_split])[0]

        return {'variant': variant, 'model_name': VARIANT_MODEL_NAMES[variant]}

    async def predict(
        self, variant: str, currency: str = "usd_rate", days: int = 1,
        user_id: Optional[str] = None, session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Считает реальный прогноз для указанного варианта и логирует его."""
        if variant == "A":
            forecast = await self.forecast_service.get_forecast(days=days, currency=currency)
        else:
            forecast = await self.forecast_service.get_persistence_baseline(days=days, currency=currency)

        if not forecast:
            raise ValueError(f"No forecast data available for currency={currency}")

        entry = forecast[-1] if isinstance(forecast, list) else forecast
        predicted_rate = float(entry["rate"])
        forecast_date = datetime.strptime(entry["date"], "%Y-%m-%d")

        self._log_prediction(user_id, session_id, variant, currency, predicted_rate, forecast_date)

        return {
            "variant": variant,
            "model_name": VARIANT_MODEL_NAMES[variant],
            "currency": currency,
            "predicted_rate": predicted_rate,
            "forecast_date": entry["date"],
            "confidence_interval": {"lower": entry.get("lower_bound"), "upper": entry.get("upper_bound")},
        }

    def _log_prediction(self, user_id, session_id, variant, currency, predicted_rate, forecast_date):
        session = self.Session()
        try:
            log_entry = ABTestLog(
                user_id=user_id or 'anonymous',
                session_id=session_id or 'unknown',
                variant=variant,
                currency=currency,
                model_type=VARIANT_MODEL_NAMES[variant],
                predicted_rate=predicted_rate,
                forecast_date=forecast_date,
                is_converted=False,
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging AB prediction: {e}")
        finally:
            session.close()

    async def update_actual_rates(self) -> int:
        """
        Обновляет фактические курсы для прогнозов, дата которых уже наступила.

        Ранее это был явный заглушка (commit() без единой строчки логики
        получения фактического курса). Теперь: для каждого прогноза с
        прошедшей forecast_date и ещё не заполненным actual_rate, ищем
        реальный курс ЦБ РФ на эту дату (или ближайшую предыдущую в
        пределах 3 дней — выходные/праздники, когда ЦБ не публикует курс)
        и записываем его.
        """
        session = self.Session()
        try:
            pending = session.query(ABTestLog).filter(
                ABTestLog.forecast_date.isnot(None),
                ABTestLog.actual_rate.is_(None),
                ABTestLog.forecast_date <= datetime.now(),
            ).all()

            if not pending:
                return 0

            df = await self.forecast_service.data_loader.load_data(365)
            if df is None or df.empty:
                return 0

            df = df.copy()
            df["date_only"] = pd.to_datetime(df["date"]).dt.date

            updated = 0
            for log in pending:
                col = log.currency if log.currency in df.columns else (
                    "usd_rate" if "usd" in (log.currency or "") else "eur_rate"
                )
                if col not in df.columns:
                    continue

                target_date = log.forecast_date.date()
                match = df[df["date_only"] == target_date]
                if match.empty:
                    # Weekend/holiday: fall back to the nearest earlier date
                    # within 3 days (CBR doesn't publish rates on those days).
                    for delta in range(1, 4):
                        match = df[df["date_only"] == target_date - timedelta(days=delta)]
                        if not match.empty:
                            break

                if not match.empty:
                    log.actual_rate = float(match.iloc[-1][col])
                    updated += 1

            session.commit()
            return updated
        except Exception as e:
            session.rollback()
            print(f"Error updating actual rates: {e}")
            return 0
        finally:
            session.close()

    def get_ab_stats(self, days: int = 30) -> Dict[str, Any]:
        """Получает статистику A/B-теста за последние N дней."""
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            logs = session.query(ABTestLog).filter(
                ABTestLog.timestamp >= cutoff_date,
            ).all()

            total_requests = len(logs)
            evaluated = [log for log in logs if log.actual_rate is not None]

            df = pd.DataFrame([{
                'variant': log.variant,
                'predicted': log.predicted_rate,
                'actual': log.actual_rate,
                'model_type': log.model_type,
            } for log in evaluated])

            stats: Dict[str, Any] = {
                "period_days": days,
                "total_requests": total_requests,
                "evaluated_requests": len(evaluated),
                "traffic_split": {"A": self.traffic_split, "B": round(1 - self.traffic_split, 4)},
            }

            if df.empty:
                stats["message"] = (
                    "Есть назначения вариантов, но пока нет прогнозов с известным "
                    "фактическим курсом (запустите scripts/update_ab_actual_rates.py "
                    "после того, как прогнозируемая дата наступит)."
                    if total_requests > 0 else
                    "Нет данных за этот период — сделайте несколько запросов к /api/ab-test/predict."
                )
                return stats

            for variant in ['A', 'B']:
                group = df[df['variant'] == variant]
                if group.empty:
                    continue
                mae = float(np.mean(np.abs(group['predicted'] - group['actual'])))
                mape = float(np.mean(np.abs((group['predicted'] - group['actual']) / group['actual'])) * 100)
                stats[f'variant_{variant}'] = {
                    'model_name': VARIANT_MODEL_NAMES[variant],
                    'count': int(len(group)),
                    'mae': round(mae, 4),
                    'mape': round(mape, 2),
                    'mean_predicted': round(float(group['predicted'].mean()), 4),
                    'mean_actual': round(float(group['actual'].mean()), 4),
                }

            if 'variant_A' in stats and 'variant_B' in stats:
                errors_a = df[df['variant'] == 'A']['predicted'] - df[df['variant'] == 'A']['actual']
                errors_b = df[df['variant'] == 'B']['predicted'] - df[df['variant'] == 'B']['actual']

                if len(errors_a) >= 2 and len(errors_b) >= 2:
                    from scipy import stats as scipy_stats
                    t_stat, p_value = scipy_stats.ttest_ind(errors_a, errors_b, equal_var=False)
                    stats['statistical_significance'] = {
                        't_statistic': round(float(t_stat), 4),
                        'p_value': round(float(p_value), 6),
                        'is_significant': bool(p_value < 0.05),
                    }

                a_mae, b_mae = stats['variant_A']['mae'], stats['variant_B']['mae']
                if a_mae < b_mae:
                    stats['recommendation'] = "Вариант A (ML-ансамбль) точнее наивного baseline — есть смысл использовать модель."
                elif b_mae < a_mae:
                    stats['recommendation'] = "Наивный baseline (B) точнее ML-ансамбля на этом периоде — модель пока не даёт преимущества."
                else:
                    stats['recommendation'] = "Точность вариантов совпадает на этом периоде."

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
