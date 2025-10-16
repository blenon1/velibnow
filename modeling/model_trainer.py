from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

class ModelTrainer:
    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path

    def train(self, df):
        X = df[["ratio_occ", "capacity", "is_holiday", "temperature_2m"]]
        y = (df["numbikesavailable"] == 0).astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model = GradientBoostingClassifier().fit(X_train, y_train)
        print("AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
        joblib.dump(model, self.model_path)
        return model




from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import joblib


class SimpleVelibModel:
    """
    Pipeline simple et robuste pour prédire si une station sera vide/pleine.
    - Prépare et standardise les variables par type
    - Encode proprement
    - Intègre des features temps (cycliques) et des flags météo simples (pluie/vent/soleil)
    - Entraîne un classifieur lisible (GB ou Logit)
    """

    # ---- colonnes attendues (si présentes) ----
    BASE_NUMERIC: List[str] = ["capacity", "docks_available", "fill_rate", "rolling_fill_rate"]
    BASE_BINARY:  List[str] = ["holiday_flag", "is_weekend", "operative"]
    BASE_CATEG:   List[str] = ["station_id"]  # effet station utile

    # météo candidates (si absentes, ignorées)
    WEATHER_NUMERIC_CAND = ["precipitation", "rain", "snowfall", "windspeed_10m", "cloudcover"]
    WEATHER_CATEG_CAND   = ["weathercode", "station_name_weather"]

    # flags météo simples qui seront ajoutés (0/1)
    SIMPLE_WEATHER_FLAGS = ["is_raining", "is_windy", "is_sunny"]

    def __init__(
        self,
        target_col: str = "target_empty",        # ou "target_full"
        model_type: str = "gb",                  # "gb" ou "logit"
        timezone: str = "Europe/Paris",
        random_state: int = 42,
        wind_speed_ms_windy: float = 8.0,        # ~29 km/h
        rain_mmph_raining: float = 0.1,          # >0.1 mm/h = pluie
        cloudcover_sunny_max: int = 30           # % de nébulosité max pour "soleil"
    ):
        self.target_col = target_col
        self.model_type = model_type
        self.timezone = timezone
        self.random_state = random_state
        self.wind_speed_ms_windy = wind_speed_ms_windy
        self.rain_mmph_raining = rain_mmph_raining
        self.cloudcover_sunny_max = cloudcover_sunny_max

        # seront détectées en fonction du DF
        self.weather_numeric_: List[str] = []
        self.weather_categ_: List[str] = []

        self.feature_list_: List[str] = []   # ordre final des features
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.pipeline_: Optional[Pipeline] = None

    # ---------- helpers: construction des features ----------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time" not in df.columns:
            raise ValueError("La colonne 'time' est requise pour générer les features temporelles.")
        ts = pd.to_datetime(df["time"], utc=True).dt.tz_convert(self.timezone)
        out = df.copy()
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.weekday     # 0=lundi
        out["month"] = ts.dt.month

        # Encodage cyclique
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"]  = np.sin(2 * np.pi * out["dow"] / 7)
        out["dow_cos"]  = np.cos(2 * np.pi * out["dow"] / 7)
        # On garde "month" en numérique (captation saison)
        return out

    def _make_simple_weather_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        precip = None
        if "precipitation" in out.columns:
            precip = pd.to_numeric(out["precipitation"], errors="coerce").fillna(0)
        elif "rain" in out.columns:
            precip = pd.to_numeric(out["rain"], errors="coerce").fillna(0)
        else:
            precip = pd.Series(0.0, index=out.index)

        snow = pd.to_numeric(out["snowfall"], errors="coerce").fillna(0) if "snowfall" in out.columns else 0.0
        wind = pd.to_numeric(out["windspeed_10m"], errors="coerce").fillna(0) if "windspeed_10m" in out.columns else 0.0
        clouds = pd.to_numeric(out["cloudcover"], errors="coerce") if "cloudcover" in out.columns else np.nan
        wcode = pd.to_numeric(out["weathercode"], errors="coerce") if "weathercode" in out.columns else pd.Series(np.nan, index=out.index)

        # pluie / neige / codes pluvieux
        raining_codes = {51,53,55,56,57,61,63,65,66,67,80,81,82}
        out["is_raining"] = (
            (precip >= self.rain_mmph_raining) |
            (snow > 0) |
            (wcode.isin(raining_codes))
        ).astype(int)

        # vent fort (m/s). Si tes vitesses sont en km/h, convertis-les avant d'appeler fit/predict.
        wind_ms = wind
        out["is_windy"] = (wind_ms >= self.wind_speed_ms_windy).astype(int)

        # ensoleillé: peu nuageux ET pas de précipitation OU codes clairs
        clear_codes = {0, 1}
        no_precip = (precip < self.rain_mmph_raining) & (pd.Series(snow, index=out.index) == 0)
        sunny_by_cloud = clouds.notna() & (clouds <= self.cloudcover_sunny_max) & no_precip
        sunny_by_code = wcode.isin(clear_codes)
        out["is_sunny"] = (sunny_by_cloud | sunny_by_code).astype(int)
        return out

    def _detect_weather_columns(self, df: pd.DataFrame) -> None:
        self.weather_numeric_ = [c for c in self.WEATHER_NUMERIC_CAND if c in df.columns]
        self.weather_categ_   = [c for c in self.WEATHER_CATEG_CAND if c in df.columns]

    def _build_preprocessor(self) -> None:
        # Numériques standardisées
        numeric_cols = (
            self.BASE_NUMERIC
            + self.BASE_BINARY
            + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month"]
            + self.weather_numeric_
            + self.SIMPLE_WEATHER_FLAGS
        )
        # Catégorielles encodées
        categ_cols = self.BASE_CATEG + self.weather_categ_

        self.feature_list_ = numeric_cols + categ_cols
        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categ_cols),
            ],
            remainder="drop",
        )

    def _prepare_features(self, df: pd.DataFrame, with_target: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        # 1) temps
        df2 = self._add_time_features(df)
        # 2) flags météo
        self._detect_weather_columns(df2)
        df2 = self._make_simple_weather_flags(df2)
        # 3) (re)construire le préprocesseur
        self._build_preprocessor()

        # 4) vérifier la présence des bases
        for col in self.BASE_NUMERIC + self.BASE_BINARY + self.BASE_CATEG:
            if col not in df2.columns:
                raise ValueError(f"Colonne manquante: '{col}'")

        X = df2[self.feature_list_]
        y = None
        if with_target:
            if self.target_col not in df2.columns:
                raise ValueError(f"Colonne cible manquante: '{self.target_col}'")
            y = df2[self.target_col].astype(int)
        return X, y

    # ---------- API publique ----------

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, verbose: bool = True) -> dict:
        """Prépare X, entraîne le modèle choisi et renvoie des métriques simples (AUC)."""
        X, y = self._prepare_features(df, with_target=True)

        # Choix du classifieur
        if self.model_type == "logit":
            clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=self.random_state)
        else:
            clf = GradientBoostingClassifier(random_state=self.random_state)

        self.pipeline_ = Pipeline(steps=[("pre", self.preprocessor_), ("clf", clf)])

        # split simple (pour rester lisible). Pour un vrai cas prod: split temporel.
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        self.pipeline_.fit(X_tr, y_tr)

        proba = self.pipeline_.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)
        metrics = {"val_auc": float(auc), "n_samples_train": int(len(X_tr)), "n_samples_val": int(len(X_va))}
        if verbose:
            print(f"[{self.target_col}] {self.model_type.upper()}  AUC={metrics['val_auc']:.3f}  "
                  f"(train={metrics['n_samples_train']}, val={metrics['n_samples_val']})")
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilité d'être positif (ex: vide si target_empty)."""
        check_is_fitted(self.pipeline_, "named_steps")
        X, _ = self._prepare_features(df, with_target=False)
        return self.pipeline_.predict_proba(X)[:, 1]

    def predict_label(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Label binaire selon un seuil."""
        p = self.predict_proba(df)
        return pd.Series((p >= threshold).astype(int), index=df.index, name=f"{self.target_col}_pred")

    def save(self, path: str) -> None:
        """Sauvegarde le pipeline complet (prétraitement + modèle)."""
        check_is_fitted(self.pipeline_, "named_steps")
        joblib.dump(self.pipeline_, path)

    def load(self, path: str) -> None:
        """Charge un pipeline entraîné."""
        self.pipeline_ = joblib.load(path)
