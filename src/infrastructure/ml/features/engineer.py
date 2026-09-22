import pandas as pd

from src.core.constants import SUPPORTED_CURRENCIES


class FeatureEngineer:
    """
    Класс для создания признаков (lags, rolling stats, calendar features) из временных рядов.
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Календарные признаки
            df["day_of_week"] = df["date"].dt.dayofweek
            df["day"] = df["date"].dt.day
            df["month"] = df["date"].dt.month

        # Лаги и скользящие показатели для каждой валютной пары, известной
        # приложению (см. src/core/constants.py) - раньше здесь были
        # захардкожены только usd_rate/eur_rate, из-за чего у CNY/GBP не
        # было бы никаких признаков для обучения модели.
        for col in SUPPORTED_CURRENCIES:
            if col in df.columns:
                for lag in [1, 2, 3, 7]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                for window in [3, 7]:
                    df[f"{col}_rolling_mean_{window}"] = df[col].shift(1).rolling(window=window).mean()
                    df[f"{col}_rolling_std_{window}"] = df[col].shift(1).rolling(window=window).std()
                # Моментум: относительное изменение курса за N дней. Лаги
                # и скользящие дают модели "уровень", но не "скорость" -
                # в финансовых рядах эти два сигнала часто ведут себя
                # по-разному (курс может стоять на месте после резкого
                # скачка, и наоборот). Сдвинуто на 1 день назад, как и
                # остальные признаки, чтобы не заглядывать в будущее.
                for horizon in [3, 7]:
                    df[f"{col}_pct_change_{horizon}"] = df[col].pct_change(horizon).shift(1)

        # Кросс-курсы: относительная сила доллара против других валют в
        # том же наборе. Если доллар одновременно дорожает против евро,
        # юаня и фунта, это сигнал "доллар крепчает глобально" - то, что
        # ни один из отдельных лагов usd_rate сам по себе не выражает.
        # Сдвинуто на 1 день назад: использование СЕГОДНЯШНЕГО
        # eur_rate/usd_rate как признака при предсказании СЕГОДНЯШНЕГО
        # usd_rate было бы прямой утечкой целевой переменной.
        cross_pairs = [
            ("eur_rate", "usd_rate"),
            ("cny_rate", "usd_rate"),
            ("gbp_rate", "usd_rate"),
        ]
        for numerator, denominator in cross_pairs:
            if numerator in df.columns and denominator in df.columns:
                df[f"{numerator}_{denominator}_cross_lag1"] = (
                    df[numerator] / df[denominator]
                ).shift(1)

        # Ключевая ставка ЦБ (если loader её подмешал - см.
        # src/infrastructure/data/loader.py и external_features.py).
        # Сама по себе публична и известна заранее на весь день вперёд
        # (ЦБ объявляет решение по ставке, а не публикует его задним
        # числом), поэтому, в отличие от курсов, сдвиг на 1 день здесь
        # не нужен - никакой утечки целевой переменной нет. Данные
        # разреженные (ставка меняется несколько раз в год) - заполняем
        # пропуски последним известным значением.
        if "key_rate" in df.columns:
            df["key_rate"] = df["key_rate"].ffill().bfill()

        return df
