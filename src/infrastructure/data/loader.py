import asyncio
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.constants import CBR_LETTER_CODES, CBR_VALUTE_IDS
from src.infrastructure.data.cache import CacheManager
from src.infrastructure.data import external_features
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.cache = CacheManager()
        self._last_update = None
        self._update_lock = asyncio.Lock()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_current_daily(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch today's exchange rates for every supported currency from CBR."""
        url = "http://www.cbr.ru/scripts/XML_daily.asp"
        rates: Dict[str, float] = {}
        try:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, "xml")

                for column, valute_id in CBR_VALUTE_IDS.items():
                    item = soup.find("Valute", {"ID": valute_id})
                    if not item:
                        continue
                    value = item.find("Value")
                    nominal = item.find("Nominal")
                    if value and nominal:
                        rates[column] = float(value.text.replace(",", ".")) / int(nominal.text)

                logger.info("Fetched current rates: %s", rates)

        except Exception as e:
            logger.error(f"Failed to fetch daily currency data: {e}")
            try:
                rates = await self._fetch_alternative_rates(session)
            except Exception as alt_e:
                logger.error(f"Alternative fetch also failed: {alt_e}")

        return rates

    async def _fetch_alternative_rates(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fallback method to fetch rates from an alternative CBR mirror."""
        url = "https://www.cbr-xml-daily.ru/daily_json.js"
        rates: Dict[str, float] = {}
        try:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                valutes = data.get("Valute", {})
                for column, letter in CBR_LETTER_CODES.items():
                    if letter in valutes:
                        entry = valutes[letter]
                        rates[column] = entry["Value"] / entry.get("Nominal", 1)
                logger.info(f"Fetched alternative rates: {rates}")
        except Exception as e:
            logger.error(f"Failed to fetch alternative rates: {e}")
        return rates

    async def load_for_period(self, days: int) -> pd.DataFrame:
        """Load historical data for a period with improved caching."""
        cache_key = f"data_{days}_{datetime.now().strftime('%Y%m%d')}"

        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            try:
                df = pd.DataFrame(cached_data)
                if len(df) > 0:
                    logger.info(f"Returning cached data for {days} days")
                    return df
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        d1 = start_date.strftime("%d/%m/%Y")
        d2 = end_date.strftime("%d/%m/%Y")

        async with aiohttp.ClientSession() as session:
            try:
                historical_by_currency = {}
                for column, valute_id in CBR_VALUTE_IDS.items():
                    historical_by_currency[column] = await self._fetch_historical(session, valute_id, d1, d2)

                current_rates = await self._fetch_current_daily(session)

                valid_data = self._combine_historical_data(historical_by_currency, current_rates)

                if valid_data and len(valid_data) > 0:
                    df = pd.DataFrame(valid_data).sort_values("date").reset_index(drop=True)

                    # Best-effort macro feature: CBR's own key interest
                    # rate (see external_features.py). Never blocks - on
                    # any failure this is just an empty dict and df is
                    # returned unchanged.
                    key_rate_by_date = await external_features.fetch_key_rate(session, start_date, end_date)
                    if key_rate_by_date:
                        kr_df = pd.DataFrame(
                            {
                                "date": pd.to_datetime(list(key_rate_by_date.keys()), format="%d.%m.%Y"),
                                "key_rate": list(key_rate_by_date.values()),
                            }
                        ).sort_values("date")
                        # "As of" join: each row gets the key rate that
                        # was in effect on that date (the most recent
                        # change at or before it), not just rows where a
                        # change happened to fall exactly on that date.
                        df = pd.merge_asof(df, kr_df, on="date", direction="backward")

                    self.cache.set(cache_key, df.to_dict("records"), 600)
                    self._last_update = datetime.now()
                    logger.info(f"Successfully loaded {len(df)} records for {days} days")
                    return df
                else:
                    logger.warning("No valid data fetched, using demo data")
                    return self._generate_demo_data(days)

            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return self._generate_demo_data(days)

    async def _fetch_historical(self, session: aiohttp.ClientSession, val_id: str, date1: str, date2: str) -> Dict[str, float]:
        """Fetch historical data for a specific currency."""
        url = f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={date1}&date_req2={date2}&VAL_NM_RQ={val_id}"
        result = {}

        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, "xml")
                    records = soup.find_all("Record")

                    for record in records:
                        date = record.get("Date")
                        value_elem = record.find("Value")
                        nominal_elem = record.find("Nominal")

                        if date and value_elem and nominal_elem:
                            try:
                                value = float(value_elem.text.replace(",", "."))
                                nominal = int(nominal_elem.text)
                                result[date] = value / nominal
                            except (ValueError, TypeError) as e:
                                logger.debug(f"Error parsing record: {e}")
                                continue
                else:
                    logger.warning(f"Failed to fetch historical data for {val_id}: status {response.status}")

        except Exception as e:
            logger.error(f"Error fetching historical data for {val_id}: {e}")

        return result

    def _combine_historical_data(self, historical_by_currency: Dict[str, Dict], current_rates: Dict) -> List[Dict]:
        """Combine every currency's historical series (keyed by CBR's own
        "dd.mm.yyyy" date strings) with today's rates, one row per date
        that has data for at least the primary pair (USD/EUR) - a
        currency missing on a given date (e.g. a CBR outage for just that
        one series) simply gets no value that day rather than dropping
        the whole row, matching the old USD-and-EUR-both-required
        behaviour when both of those are present.
        """
        valid_data = []

        primary_dates = set(historical_by_currency.get("usd_rate", {}).keys()) & \
            set(historical_by_currency.get("eur_rate", {}).keys())

        for date_str in sorted(primary_dates):
            try:
                date = datetime.strptime(date_str, "%d.%m.%Y")
            except ValueError:
                continue
            row = {"date": date}
            for column, series in historical_by_currency.items():
                if date_str in series:
                    row[column] = round(series[date_str], 4)
            valid_data.append(row)

        if current_rates:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            has_today = any(record["date"] == today for record in valid_data)

            if not has_today:
                row = {"date": today, **{c: round(v, 4) for c, v in current_rates.items()}}
                valid_data.append(row)
                logger.info(f"Added today's rates: {current_rates}")
            else:
                for record in valid_data:
                    if record["date"] == today:
                        record.update({c: round(v, 4) for c, v in current_rates.items()})
                        logger.info(f"Updated today's rates: {current_rates}")
                        break
        else:
            logger.warning("No current rates available to add/update")

        return valid_data

    def _generate_demo_data(self, days: int) -> pd.DataFrame:
        """Generate demo data as fallback, when CBR is unreachable."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days + 1)]

        np.random.seed(42)
        # Roughly realistic starting points per currency (RUB per unit,
        # CNY per 10 units to match the CBR convention).
        base_levels = {"usd_rate": 75.0, "eur_rate": 82.0, "cny_rate": 105.0, "gbp_rate": 95.0}

        data = {"date": dates}
        for column, base in base_levels.items():
            walk = base + np.random.normal(0, 0.5, len(dates))
            cumsum = np.cumsum(walk)
            data[column] = np.round(cumsum / np.arange(1, len(dates) + 1), 4)

        return pd.DataFrame(data)

    async def load_data(self, days: int = 90) -> pd.DataFrame:
        """Main method to load data with retry logic."""
        try:
            df = await self.load_for_period(days)
            if df is None or len(df) == 0:
                logger.warning("No data loaded, using demo data")
                df = self._generate_demo_data(days)
            return df
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            return self._generate_demo_data(days)

    async def refresh_data(self, days: int = 365) -> pd.DataFrame:
        """Force refresh data from source."""
        cache_key = f"data_{days}_{datetime.now().strftime('%Y%m%d')}"
        self.cache.delete(cache_key)
        return await self.load_for_period(days)
