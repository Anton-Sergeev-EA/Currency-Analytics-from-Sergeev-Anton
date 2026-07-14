from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Currency:
    code: str
    name: str
    symbol: str

    def __str__(self) -> str:
        return self.code


@dataclass
class ExchangeRate:
    currency: Currency
    rate: float
    date: datetime
    nominal: int = 1

    def to_rub(self, amount: float) -> float:
        return amount * self.rate / self.nominal

    def to_dict(self) -> dict:
        return {
            "currency": self.currency.code,
            "rate": self.rate,
            "date": self.date.isoformat(),
            "nominal": self.nominal
        }
