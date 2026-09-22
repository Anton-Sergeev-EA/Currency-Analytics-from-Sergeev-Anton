"""
ExternalFeaturesLoader — macro series that actually move the ruble, to give
the ensemble something beyond its own past values to learn from.

Forecasting a currency from nothing but its own lags is close to predicting
a random walk from its own noise (see model_evaluation.py's backtest,
which came back with a negative R2 on lag-only features). This module adds
one genuinely external source: the Bank of Russia's own key interest rate,
which is one of the few macro variables with a documented, no-API-key CBR
endpoint (the same family as XML_dynamic.asp/XML_daily.asp in loader.py,
just a SOAP method instead of a plain query-string GET).

Every fetch here is strictly best-effort: on any failure (network,
unexpected response shape, CBR changing their schema) it logs a warning
and returns an empty dict, and the caller simply doesn't get that feature
- it must never take model training down. This module's own network
access could not be independently verified end-to-end from the sandboxed
environment it was written in (only CBR's plain XML_*.asp endpoints
already used by loader.py could be); the response-shape parsing below is
deliberately schema-tolerant for that reason, and train_models.py prints
whether it actually got a nonzero key rate series so this is easy to
confirm on a real deployment.
"""
import logging
from datetime import datetime
from typing import Dict

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# CBR's DailyInfoWebServ SOAP service - documented at
# https://www.cbr.ru/development/DWS/ - KeyRate(fromDate, ToDate) returns
# a small dataset with one row per date the key rate *changed* (it's not
# a daily series - it only moves a handful of times a year), so this is
# naturally sparse. Callers merge it onto a daily frame with an as-of
# ("effective as of this date") join, not a plain equi-join.
_KEY_RATE_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
_KEY_RATE_ENVELOPE = """<?xml version="1.0" encoding="utf-8"?>
<soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                  xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
  <soap12:Body>
    <KeyRate xmlns="http://web.cbr.ru/">
      <fromDate>{from_date}</fromDate>
      <ToDate>{to_date}</ToDate>
    </KeyRate>
  </soap12:Body>
</soap12:Envelope>"""


async def fetch_key_rate(
    session: aiohttp.ClientSession, start: datetime, end: datetime
) -> Dict[str, float]:
    """Returns {"dd.mm.yyyy": key_rate_percent} for every date the rate
    changed within [start, end] - sparse by construction. Never raises;
    returns {} on any failure so training can proceed without this
    feature rather than crashing.
    """
    body = _KEY_RATE_ENVELOPE.format(
        from_date=start.strftime("%Y-%m-%d"), to_date=end.strftime("%Y-%m-%d")
    )
    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
    rates: Dict[str, float] = {}
    try:
        async with session.post(
            _KEY_RATE_URL, data=body.encode("utf-8"), headers=headers, timeout=15
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            soup = BeautifulSoup(text, "xml")

            # Schema-tolerant: rather than assume exact row/column tag
            # names (unverifiable from here), scan every element for a
            # child pair that looks like (ISO-ish date, plausible
            # percentage) and treat that as one key-rate observation.
            for elem in soup.find_all():
                children = elem.find_all(recursive=False)
                if len(children) < 2:
                    continue
                date_val = None
                rate_val = None
                for child in children:
                    raw = (child.text or "").strip()
                    if not raw:
                        continue
                    if date_val is None:
                        try:
                            date_val = datetime.fromisoformat(raw.split("T")[0])
                            continue
                        except ValueError:
                            pass
                    if rate_val is None:
                        try:
                            candidate = float(raw.replace(",", "."))
                            if 0 < candidate < 100:  # sane bound for a % rate
                                rate_val = candidate
                        except ValueError:
                            pass
                if date_val is not None and rate_val is not None:
                    rates[date_val.strftime("%d.%m.%Y")] = rate_val

        if rates:
            logger.info("Fetched %d CBR key rate change(s)", len(rates))
        else:
            logger.warning(
                "CBR key rate response parsed but yielded 0 rows - "
                "response shape may not match what this parser expects; "
                "key_rate feature will be skipped for this run."
            )
    except Exception as exc:
        logger.warning("Could not fetch CBR key rate (feature will be skipped): %s", exc)

    return rates
