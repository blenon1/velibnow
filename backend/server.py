# backend/server.py
from __future__ import annotations

import os
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import Predictor

# --------- Config via env ---------
MODEL_EMPTY_PATH = os.getenv("MODEL_EMPTY_PATH", "/models/model_empty.joblib")
MODEL_FULL_PATH  = os.getenv("MODEL_FULL_PATH",  "/models/model_full.joblib")
STATIONS_CSV     = os.getenv("STATIONS_CSV", "/data/stations.csv")

API_STATIONS = (
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
)

app = FastAPI(title="Velib Backend", version="1.0.0")

# CORS large en dev (restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Utils data ---------
def fetch_stations_api() -> pd.DataFrame:
    r = requests.get(API_STATIONS, timeout=20)
    r.raise_for_status()
    stations = r.json()["data"]["stations"]
    df = pd.DataFrame(stations)
    return df[["station_id", "name", "capacity", "lat", "lon"]]

def read_station_names_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV introuvable: {path}")
    df = pd.read_csv(path)
    if "station_name" in df.columns:
        names = df["station_name"]
    elif "name" in df.columns:
        names = df["name"]
    else:
        raise ValueError("Le CSV doit contenir une colonne 'station_name' ou 'name'")
    return (
        names.dropna().astype(str).drop_duplicates().sort_values().tolist()
    )

def build_model_input_from_csv(
    df_api: pd.DataFrame,
    csv_names: List[str],
    utc_time_iso: str,
    holiday_flag: bool,
    is_weekend: bool,
    operative: bool,
    pluie: int, vent: int, soleil: int, nuage: int,
) -> pd.DataFrame:
    # Filtre API sur les noms du CSV
    df = df_api[df_api["name"].astype(str).isin(csv_names)].copy()
    if df.empty:
        raise ValueError("Aucune correspondance API pour les station_name du CSV.")

    # Colonne catégorielle pour le modèle
    df["station_name"] = df["name"].astype(str)

    # Flags & time
    df["time"] = pd.to_datetime(utc_time_iso, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    df["holiday_flag"] = int(bool(holiday_flag))
    df["is_weekend"]   = int(bool(is_weekend))
    df["operative"]    = int(bool(operative))
    df["pluie"]  = int(bool(pluie))
    df["vent"]   = int(bool(vent))
    df["soleil"] = int(bool(soleil))
    df["nuage"]  = int(bool(nuage))

    # Colonnes finales nécessaires au modèle + méta
    keep = [
        "time",
        "holiday_flag", "is_weekend", "operative",
        "pluie", "vent", "soleil", "nuage",
        "station_name",           # catégorielle
        "station_id", "name", "capacity", "lat", "lon",  # méta pour affichage
    ]
    return df[keep]

# --------- Modèles préchargés (catégorielle = station_name) ---------
try:
    pred_depart = Predictor(categ_col="station_name", target_col="target_empty")
    pred_depart.load(MODEL_EMPTY_PATH)

    pred_arrive = Predictor(categ_col="station_name", target_col="target_full")
    pred_arrive.load(MODEL_FULL_PATH)
except Exception as e:
    raise RuntimeError(f"Impossible de charger les modèles: {e}")

# --------- Endpoints ---------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/stations")
def stations_api() -> Dict[str, Any]:
    """Renvoie les stations de l'API Vélib (informatif pour le front)."""
    try:
        df = fetch_stations_api()
        return {"count": int(len(df)), "stations": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(502, f"Erreur API Vélib: {e}")

@app.get("/stations/csv")
def stations_from_csv() -> Dict[str, Any]:
    """Renvoie la liste des station_name (ou name) chargés depuis le CSV côté backend."""
    try:
        names = read_station_names_from_csv(STATIONS_CSV)
        return {"count": len(names), "station_names": names}
    except Exception as e:
        raise HTTPException(400, f"Erreur CSV: {e}")

@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Payload minimal:
    {
      "utc_time_iso": "...",
      "holiday_flag": true/false,
      "is_weekend": true/false,
      "operative": true/false,
      "pluie": 0/1, "vent": 0/1, "soleil": 0/1, "nuage": 0/1
    }
    -> Utilise la liste de station_name provenant du CSV.
    """
    try:
        # Lire flags
        utc_time_iso = payload["utc_time_iso"]
        holiday_flag = payload.get("holiday_flag", False)
        is_weekend   = payload.get("is_weekend", False)
        operative    = payload.get("operative", True)
        pluie  = int(payload.get("pluie", 0))
        vent   = int(payload.get("vent", 0))
        soleil = int(payload.get("soleil", 1))
        nuage  = int(payload.get("nuage", 0))

        # Données sources
        df_api = fetch_stations_api()
        names  = read_station_names_from_csv(STATIONS_CSV)

        # Construction entrée modèle
        df_in = build_model_input_from_csv(
            df_api, names, utc_time_iso, holiday_flag, is_weekend, operative, pluie, vent, soleil, nuage
        )

        # Prédictions
        proba_dep = pred_depart.predict_proba(df_in)
        proba_arr = pred_arrive.predict_proba(df_in)

        # Assemblage résultat (on renvoie méta + proba)
        out = df_in[["station_name", "station_id", "name", "capacity", "lat", "lon"]].copy()
        out["proba_velo_depart"]   = proba_dep
        out["proba_place_arrivee"] = proba_arr

        return {"count": int(len(out)), "rows": out.to_dict(orient="records")}
    except KeyError as e:
        raise HTTPException(400, f"Champ requis manquant: {e}")
    except Exception as e:
        raise HTTPException(400, f"Erreur prédiction: {e}")
