# backend/predictor.py
from __future__ import annotations
from io import BytesIO
from typing import List
import numpy as np
import pandas as pd
import joblib
from sklearn.utils.validation import check_is_fitted


class Predictor:
    """
    - Supporte station_id OU station_name (choisi via categ_col)
    - load(path) / load_bytes(raw)
    - predict_proba(df)
    Schéma attendu: time (UTC ISO), flags (holiday_flag, is_weekend, operative,
    pluie, vent, soleil, nuage) + colonne catégorielle (station_id ou station_name).
    """

    def __init__(self, categ_col: str = "station_name", target_col: str = "target_empty"):
        if categ_col not in {"station_id", "station_name"}:
            raise ValueError("categ_col doit être 'station_id' ou 'station_name'")
        self.categ_col = categ_col
        self.target_col = target_col
        self.pipeline_ = None
        self.preprocessor_ = None

        base_time = ["hour_ssin", "hour_ccos", "dow_sin", "dow_cos", "month"]
        flags = ["holiday_flag", "is_weekend", "operative", "pluie", "vent", "soleil", "nuage"]
        self.feature_list_: List[str] = base_time + flags + [self.categ_col]

    # ---------- features ----------
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["time"], utc=True)
        out = df.copy()
        hour = ts.dt.hour
        dow = ts.dt.weekday
        out["month"] = ts.dt.month
        out["hour_ssin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_ccos"] = np.cos(2 * np.pi * hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        return out

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = self._add_time_features(df)
        missing = [c for c in self.feature_list_ if c not in df2.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes pour la prédiction: {missing}")
        return df2[self.feature_list_]

    # ---------- I/O ----------
    def load(self, path: str) -> None:
        self.pipeline_ = joblib.load(path)
        self.preprocessor_ = getattr(self.pipeline_, "named_steps", {}).get("preprocessor", None)

    def load_bytes(self, raw_bytes: bytes) -> None:
        if not raw_bytes:
            raise ValueError("Fichier modèle vide (0 octet).")
        self.pipeline_ = joblib.load(BytesIO(raw_bytes))
        self.preprocessor_ = getattr(self.pipeline_, "named_steps", {}).get("preprocessor", None)

    # ---------- API prédiction ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self.pipeline_, "named_steps")
        X = self._prepare_features(df)
        return self.pipeline_.predict_proba(X)[:, 1]
