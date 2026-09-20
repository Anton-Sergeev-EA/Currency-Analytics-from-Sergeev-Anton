"""
FinanceAdvisor — deterministic financial calculations for the RAG assistant.

Small local LLMs (this project defaults to `tinyllama`, chosen specifically
because it is cheap enough to run alongside the rest of the app on a
4GB-RAM VDS) are unreliable at arithmetic. Asking one to compute "how much
will I earn if I invest 100,000 RUB in EUR for 2 weeks" and quoting its
answer as fact would be presenting a guess as a calculation.

Instead, questions that ask for a number - unit conversion, an investment
projection, a currency comparison - are parsed here with plain regular
expressions and answered with an exact calculation from the real current
rate and the ML forecast, no LLM involved. The LLM is reserved for what it
is actually good at: turning retrieved knowledge-base text into a natural
-language answer for open-ended questions (see RAGService).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

CURRENCY_ALIASES = {
    "usd": ["доллар", "доллара", "долларов", "usd", "$", "бакс", "баксов"],
    "eur": ["евро", "eur", "€"],
    "rub": ["рубл", "руб", "rub", "₽"],
}

_AMOUNT_RE = re.compile(r"(\d[\d\s]*(?:[.,]\d+)?)\s*(?:тыс\.?|тысяч[аи]?)?", re.IGNORECASE)
_DAYS_RE = re.compile(r"(\d{1,3})\s*(?:дн[еяй]|day|days|недел)", re.IGNORECASE)
_WEEKS_RE = re.compile(r"(\d{1,2})\s*недел", re.IGNORECASE)
_MONTHS_RE = re.compile(r"(\d{1,2})\s*месяц", re.IGNORECASE)


def _detect_currency(text: str, default: Optional[str] = None) -> Optional[str]:
    text_l = text.lower()
    for code, aliases in CURRENCY_ALIASES.items():
        if any(alias in text_l for alias in aliases):
            return code
    return default


def _parse_amount(text: str) -> Optional[float]:
    """Extracts the largest plausible amount mentioned in the question."""
    candidates = []
    for match in _AMOUNT_RE.finditer(text):
        raw = match.group(1).replace(" ", "").replace(",", ".")
        if not raw or raw == ".":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        # "тыс"/"тысяч" right after the number means multiply by 1000
        # (e.g. "100 тысяч рублей").
        tail = text[match.end(): match.end() + 12].lower()
        if "тыс" in match.group(0).lower() or "тыс" in tail:
            value *= 1000
        candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _parse_horizon_days(text: str, default_days: int = 7) -> int:
    text_l = text.lower()
    week_match = _WEEKS_RE.search(text_l)
    if week_match:
        return min(int(week_match.group(1)) * 7, 30)
    month_match = _MONTHS_RE.search(text_l)
    if month_match:
        return min(int(month_match.group(1)) * 30, 30)
    day_match = re.search(r"(\d{1,3})\s*дн", text_l)
    if day_match:
        return min(int(day_match.group(1)), 30)
    if "недел" in text_l:
        return 7
    if "месяц" in text_l:
        return 30
    return default_days


class FinanceAdvisor:
    """Detects finance-calculation intents and answers them deterministically."""

    INVESTMENT_TRIGGERS = ["заработа", "вложу", "вложить", "инвестир", "прибыл", "доход"]
    CONVERT_TRIGGERS = ["переведи", "конверт", "сколько будет", "сколько рублей", "сколько стоит",
                         "перевести", "в рублях", "в долларах", "в евро"]
    COMPARE_TRIGGERS = ["сравни", "что лучше", "какая валюта лучше", "выгодн"]
    FORECAST_TRIGGERS = ["прогноз", "прогнозу", "вырастет", "упадет", "упадёт", "динамик", "куда пойдет", "куда пойдёт"]

    def detect_intent(self, question: str) -> str:
        q = question.lower()
        if any(t in q for t in self.INVESTMENT_TRIGGERS) and _parse_amount(q) is not None:
            return "investment"
        if any(t in q for t in self.COMPARE_TRIGGERS):
            return "comparison"
        if any(t in q for t in self.CONVERT_TRIGGERS) and _parse_amount(q) is not None:
            return "conversion"
        if any(t in q for t in self.FORECAST_TRIGGERS):
            return "forecast"
        return "general"

    def compute_forecast(
        self, question: str, current_rates: Dict[str, float], forecasts: Dict[str, list]
    ) -> Optional[Dict[str, Any]]:
        q = question.lower()
        wants_usd = _detect_currency(q) == "usd" or "доллар" in q
        wants_eur = _detect_currency(q) == "eur" or "евро" in q
        currencies = []
        if wants_usd and not wants_eur:
            currencies = ["usd"]
        elif wants_eur and not wants_usd:
            currencies = ["eur"]
        else:
            currencies = ["usd", "eur"]

        horizon = _parse_horizon_days(question)
        result = {}
        for ccy in currencies:
            forecast_list = forecasts.get(ccy) or []
            if not forecast_list or ccy not in current_rates:
                continue
            target_entry = min(forecast_list, key=lambda item: abs(item.get("day", 1) - horizon))
            forecast_rate = target_entry.get("rate") or target_entry.get("forecast")
            if forecast_rate is None:
                continue
            change_pct = ((forecast_rate - current_rates[ccy]) / current_rates[ccy]) * 100
            result[ccy] = {
                "current_rate": current_rates[ccy],
                "forecast_rate": round(float(forecast_rate), 2),
                "horizon_days": target_entry.get("day", horizon),
                "change_pct": round(change_pct, 2),
                "lower_bound": target_entry.get("lower_bound"),
                "upper_bound": target_entry.get("upper_bound"),
            }
        return result or None

    def compute_conversion(self, question: str, rates: Dict[str, float]) -> Optional[Dict[str, Any]]:
        amount = _parse_amount(question)
        if amount is None:
            return None

        from_ccy = _detect_currency(question)
        if from_ccy is None:
            return None

        if from_ccy == "rub":
            # RUB -> the other currency mentioned, defaulting to USD.
            to_ccy = "eur" if "евро" in question.lower() and "доллар" not in question.lower() else "usd"
            rate = rates.get(to_ccy)
            if not rate:
                return None
            result = amount / rate
            return {
                "amount": amount, "from": "RUB", "to": to_ccy.upper(),
                "rate": rate, "result": round(result, 2),
            }

        rate = rates.get(from_ccy)
        if not rate:
            return None
        result = amount * rate
        return {
            "amount": amount, "from": from_ccy.upper(), "to": "RUB",
            "rate": rate, "result": round(result, 2),
        }

    def compute_investment(
        self, question: str, current_rates: Dict[str, float], forecasts: Dict[str, list]
    ) -> Optional[Dict[str, Any]]:
        amount = _parse_amount(question)
        if amount is None:
            return None

        target_ccy = _detect_currency(question, default="usd")
        if target_ccy == "rub":
            target_ccy = "usd"

        current_rate = current_rates.get(target_ccy)
        forecast_list = forecasts.get(target_ccy) or []
        if not current_rate or not forecast_list:
            return None

        horizon = _parse_horizon_days(question)
        # Pick the forecast entry closest to the requested horizon.
        target_entry = min(forecast_list, key=lambda item: abs(item.get("day", 1) - horizon))
        forecast_rate = target_entry.get("rate") or target_entry.get("forecast")
        if not forecast_rate:
            return None

        units_bought = amount / current_rate
        future_value_rub = units_bought * forecast_rate
        profit = future_value_rub - amount
        profit_pct = (profit / amount) * 100 if amount else 0.0

        return {
            "amount": amount,
            "currency": target_ccy.upper(),
            "current_rate": current_rate,
            "forecast_rate": round(float(forecast_rate), 2),
            "horizon_days": target_entry.get("day", horizon),
            "future_value_rub": round(future_value_rub, 2),
            "profit_rub": round(profit, 2),
            "profit_pct": round(profit_pct, 2),
            "lower_bound": target_entry.get("lower_bound"),
            "upper_bound": target_entry.get("upper_bound"),
        }

    def compute_comparison(
        self, current_rates: Dict[str, float], forecasts: Dict[str, list]
    ) -> Optional[Dict[str, Any]]:
        result = {}
        for ccy in ("usd", "eur"):
            forecast_list = forecasts.get(ccy) or []
            if not forecast_list or ccy not in current_rates:
                continue
            last_entry = forecast_list[-1]
            forecast_rate = last_entry.get("rate") or last_entry.get("forecast")
            if forecast_rate is None:
                continue
            change_pct = ((forecast_rate - current_rates[ccy]) / current_rates[ccy]) * 100
            result[ccy] = {
                "current_rate": current_rates[ccy],
                "forecast_rate": round(float(forecast_rate), 2),
                "horizon_days": last_entry.get("day", len(forecast_list)),
                "change_pct": round(change_pct, 2),
            }
        return result or None
