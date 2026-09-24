"""
Regression test for the 500 that /api/data/data threw on the site's own
default 180-day window: merge_asof's backward join for key_rate (see
loader.py) leaves NaN for any row older than that feature's coverage, and
Starlette's JSONResponse (allow_nan=False by default) refuses to serialize
a bare NaN anywhere in the payload - not just in that one field, the whole
response 500'd. This is why the homepage's forecast chart, which requests
180 days by default, never rendered even though a 30-day request (all
recent rows, all covered by key_rate) worked fine.
"""
import asyncio
import json
import math

import pandas as pd
import pytest
from fastapi.encoders import jsonable_encoder

from src.application.services.data_service import DataService


class _FakeLoader:
    """Stands in for DataLoader.load_data(): returns a DataFrame with a
    NaN key_rate on its oldest row, exactly like a real long-period
    window whose start predates the key_rate feature's own history."""

    async def load_data(self, period_days):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "usd_rate": [80.0, 80.5, 81.0],
                "eur_rate": [90.0, 90.5, 91.0],
                "cny_rate": [11.0, 11.1, 11.2],
                "gbp_rate": [105.0, 105.5, 106.0],
                "key_rate": [float("nan"), 14.0, 14.0],
            }
        )


def test_get_historical_data_survives_a_nan_key_rate_and_stays_json_safe():
    service = DataService()
    service.loader = _FakeLoader()

    result = asyncio.run(service.get_historical_data(180))

    # The bug: `records`/`data` used to carry a bare float NaN straight
    # through from df.to_dict(), which json.dumps(..., allow_nan=False)
    # (Starlette's default) refuses to serialize.
    # jsonable_encoder is what FastAPI itself runs before Starlette's
    # JSONResponse (allow_nan=False) serializes the response - it turns
    # the Timestamp into a plain string but leaves a bare NaN/Infinity
    # float exactly as-is, which is the part this fix has to handle.
    json.dumps(jsonable_encoder(result), allow_nan=False)

    assert result["records"][0]["key_rate"] is None
    assert result["data"][0]["key_rate"] is None
    # The other two rows' real values must survive untouched.
    assert result["records"][1]["key_rate"] == 14.0
    assert result["records"][2]["key_rate"] == 14.0


def test_get_historical_data_also_cleans_an_infinite_value():
    service = DataService()
    service.loader = _FakeLoader()

    async def _load(period_days):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "usd_rate": [float("inf")],
                "eur_rate": [90.0],
                "cny_rate": [11.0],
                "gbp_rate": [105.0],
            }
        )
        return df

    service.loader.load_data = _load

    result = asyncio.run(service.get_historical_data(180))
    # jsonable_encoder is what FastAPI itself runs before Starlette's
    # JSONResponse (allow_nan=False) serializes the response - it turns
    # the Timestamp into a plain string but leaves a bare NaN/Infinity
    # float exactly as-is, which is the part this fix has to handle.
    json.dumps(jsonable_encoder(result), allow_nan=False)
    assert result["records"][0]["usd_rate"] is None
