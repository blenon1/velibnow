# dashboard/data_api.py
from __future__ import annotations
import os
import requests
import pandas as pd


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def _check(resp: requests.Response):
    resp.raise_for_status()
    return resp.json()


def get_stations_from_backend() -> pd.DataFrame:
    """Appelle GET /stations (informatif)."""
    r = requests.get(f"{BACKEND_URL}/stations", timeout=20)
    data = _check(r)
    return pd.DataFrame(data.get("stations", []))


def get_station_names_from_csv_backend() -> list[str]:
    """Appelle GET /stations/csv -> liste station_name depuis le backend."""
    r = requests.get(f"{BACKEND_URL}/stations/csv", timeout=20)
    data = _check(r)
    return data.get("station_names", [])


def predict_remote_backend(
    utc_time_iso: str,
    holiday_flag: bool,
    is_weekend: bool,
    operative: bool,
    pluie: int, vent: int, soleil: int, nuage: int,
) -> pd.DataFrame:
    """Appelle POST /predict avec les flags/heure; le backend fait tout (CSV + features + modèles)."""
    payload = {
        "utc_time_iso": utc_time_iso,
        "holiday_flag": bool(holiday_flag),
        "is_weekend": bool(is_weekend),
        "operative": bool(operative),
        "pluie": int(pluie),
        "vent": int(vent),
        "soleil": int(soleil),
        "nuage": int(nuage),
    }
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
    data = _check(r)
    return pd.DataFrame(data.get("rows", []))
