from enum import Enum

class Currency(str, Enum):
    USD = "usd_rate"
    EUR = "eur_rate"
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
    