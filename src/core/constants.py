"""
Currency registry - the single source of truth for which currencies this
app knows about.

Previously USD and EUR were hardcoded independently in half a dozen
places (loader.py's daily/historical fetchers, forecast_service.py's
target-currency branching, data_service.py, the knowledge-base builder,
finance_advisor.py's alias lists...) while this very enum sat unused
(`Currency.USD`/`Currency.EUR` were never referenced anywhere in the
codebase - grep confirms it). Adding a currency meant hunting down every
hardcoded "usd"/"eur" pair; now it means adding one entry here.

CNY and GBP were added alongside USD/EUR - the Bank of Russia publishes
both for free, and they cost nothing extra in RAM (each is a handful of
small tree-based regressors, not a separate service).
"""
from enum import Enum


class Currency(str, Enum):
    USD = "usd_rate"
    EUR = "eur_rate"
    CNY = "cny_rate"
    GBP = "gbp_rate"
    ALL = "all"


class ModelType(str, Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class DataSource(str, Enum):
    CBR = "cbr"
    DEMO = "demo"
    CACHE = "cache"


# CBR's XML_daily.asp / XML_dynamic.asp identify currencies by an internal
# "Valute ID" (unrelated to the ISO letter code). CNY's nominal is 10 (CBR
# quotes per 10 CNY, not per 1) - the loader already divides by whatever
# <Nominal> the XML gives it, so that isn't special-cased here.
CBR_VALUTE_IDS: dict[str, str] = {
    "usd_rate": "R01235",
    "eur_rate": "R01239",
    "cny_rate": "R01375",
    "gbp_rate": "R01035",
}

# Matches the fields cbr-xml-daily.ru's fallback JSON keys by (USD, EUR, ...).
CBR_LETTER_CODES: dict[str, str] = {
    "usd_rate": "USD",
    "eur_rate": "EUR",
    "cny_rate": "CNY",
    "gbp_rate": "GBP",
}

SUPPORTED_CURRENCIES: list[str] = list(CBR_VALUTE_IDS.keys())  # ["usd_rate", "eur_rate", "cny_rate", "gbp_rate"]


def short_code(column: str) -> str:
    """"usd_rate" -> "usd" """
    return column.replace("_rate", "")


def column_name(short: str) -> str:
    """"usd" -> "usd_rate" """
    short = short.lower().strip()
    return short if short.endswith("_rate") else f"{short}_rate"


# How much CBR history to train and backtest on. Shared between
# train_models.py and model_evaluation.py's backtest so the two don't
# silently disagree on what "the model" was trained on (train_models.py
# used to request 90 days while the backtest fetched 365 - two different
# training sets under one name). ~3 years of daily rates gives each
# currency's model several hundred clean rows after the feature
# engineer's lag/rolling warm-up, which four separate tree ensembles
# actually need.
DEFAULT_TRAINING_WINDOW_DAYS = 1095
