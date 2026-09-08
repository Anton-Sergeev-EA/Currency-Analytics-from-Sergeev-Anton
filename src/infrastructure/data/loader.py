import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from src.infrastructure.data.cache import CacheManager
import logging
import json
import re

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.cache = CacheManager()
        self._last_update = None
        self._update_lock = asyncio.Lock()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_current_daily(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fetch current exchange rates from CBR with proper error handling."""
        url = "http://www.cbr.ru/scripts/XML_daily.asp"
        rates = {}
        try:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, "xml")
                
                # USD - R01235
                usd_item = soup.find("Valute", {"ID": "R01235"})
                if usd_item:
                    value = usd_item.find("Value")
                    nominal = usd_item.find("Nominal")
                    if value and nominal:
                        rates["usd_rate"] = float(value.text.replace(",", ".")) / int(nominal.text)
                
                # EUR - R01239
                eur_item = soup.find("Valute", {"ID": "R01239"})
                if eur_item:
                    value = eur_item.find("Value")
                    nominal = eur_item.find("Nominal")
                    if value and nominal:
                        rates["eur_rate"] = float(value.text.replace(",", ".")) / int(nominal.text)
                
                logger.info(f"Fetched current rates: USD={rates.get('usd_rate')}, EUR={rates.get('eur_rate')}")
                
        except Exception as e:
            logger.error(f"Failed to fetch daily currency data: {e}")
            # Try alternative URL if main fails
            try:
                rates = await self._fetch_alternative_rates(session)
            except Exception as alt_e:
                logger.error(f"Alternative fetch also failed: {alt_e}")
        
        return rates

    async def _fetch_alternative_rates(self, session: aiohttp.ClientSession) -> Dict[str, float]:
        """Fallback method to fetch rates from alternative CBR endpoint."""
        url = "https://www.cbr-xml-daily.ru/daily_json.js"
        rates = {}
        try:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                if "Valute" in data:
                    if "USD" in data["Valute"]:
                        rates["usd_rate"] = data["Valute"]["USD"]["Value"]
                    if "EUR" in data["Valute"]:
                        rates["eur_rate"] = data["Valute"]["EUR"]["Value"]
                logger.info(f"Fetched alternative rates: {rates}")
        except Exception as e:
            logger.error(f"Failed to fetch alternative rates: {e}")
        return rates

    async def load_for_period(self, days: int) -> pd.DataFrame:
        """Load historical data for a period with improved caching."""
        cache_key = f"data_{days}_{datetime.now().strftime('%Y%m%d')}"
        
        # Try cache with shorter TTL
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
        
        # Format dates for CBR API
        d1 = start_date.strftime("%d/%m/%Y")
        d2 = end_date.strftime("%d/%m/%Y")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Fetch historical data
                usd_hist = await self._fetch_historical(session, "R01235", d1, d2)
                eur_hist = await self._fetch_historical(session, "R01239", d1, d2)
                
                # Fetch current rates
                current_rates = await self._fetch_current_daily(session)
                
                # Combine data
                valid_data = self._combine_historical_data(usd_hist, eur_hist, current_rates)
                
                if valid_data and len(valid_data) > 0:
                    df = pd.DataFrame(valid_data).sort_values("date").reset_index(drop=True)
                    
                    # Cache with shorter TTL (10 minutes for real-time data)
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

    def _combine_historical_data(self, usd_data: Dict, eur_data: Dict, current_rates: Dict) -> List[Dict]:
        """Combine USD and EUR historical data with current rates."""
        valid_data = []
        
        # Find common dates
        common_dates = set(usd_data.keys()) & set(eur_data.keys())
        
        for date_str in sorted(common_dates):
            try:
                date = datetime.strptime(date_str, "%d.%m.%Y")
                valid_data.append({
                    "date": date,
                    "usd_rate": round(usd_data[date_str], 4),
                    "eur_rate": round(eur_data[date_str], 4)
                })
            except ValueError:
                continue
        
        # Add current rates if available and not already in data
        if current_rates and "usd_rate" in current_rates and "eur_rate" in current_rates:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if today's data is already present
            has_today = any(record["date"] == today for record in valid_data)
            
            if not has_today:
                valid_data.append({
                    "date": today,
                    "usd_rate": round(current_rates["usd_rate"], 4),
                    "eur_rate": round(current_rates["eur_rate"], 4)
                })
                logger.info(f"Added today's rates: USD={current_rates['usd_rate']}, EUR={current_rates['eur_rate']}")
            else:
                # Update today's data with current rates
                for record in valid_data:
                    if record["date"] == today:
                        record["usd_rate"] = round(current_rates["usd_rate"], 4)
                        record["eur_rate"] = round(current_rates["eur_rate"], 4)
                        logger.info(f"Updated today's rates: USD={current_rates['usd_rate']}, EUR={current_rates['eur_rate']}")
                        break
        else:
            logger.warning("No current rates available to add/update")
        
        return valid_data

    def _generate_demo_data(self, days: int) -> pd.DataFrame:
        """Generate demo data as fallback."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days + 1)]
        
        np.random.seed(42)
        # Generate more realistic demo data
        usd_base = 75.0 + np.random.normal(0, 0.5, len(dates))
        eur_base = 82.0 + np.random.normal(0, 0.5, len(dates))
        
        usd_cumsum = np.cumsum(usd_base)
        eur_cumsum = np.cumsum(eur_base)
        
        return pd.DataFrame({
            "date": dates,
            "usd_rate": np.round(usd_cumsum / np.arange(1, len(dates) + 1), 4),
            "eur_rate": np.round(eur_cumsum / np.arange(1, len(dates) + 1), 4)
        })

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
