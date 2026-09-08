from dataclasses import dataclass


@dataclass
class Money:
    amount: float
    currency: str

    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError(f"Cannot add different currencies: {self.currency} != {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract different currencies: {self.currency} != {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: float) -> 'Money':
        return Money(self.amount * factor, self.currency)

    def __truediv__(self, divisor: float) -> 'Money':
        return Money(self.amount / divisor, self.currency)
