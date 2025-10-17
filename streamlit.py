# streamlit_app.py
from __future__ import annotations

# ========= Imports =========
import streamlit as st
import pandas as pd
import numpy as np
import requests
import tempfile
from io import BytesIO
from datetime import datetime, date, time
import pytz
from sklearn.utils.validation import check_is_fitted
import joblib

# ========= Config de page =========
st.set_page_config(
    page_title="Prédictions Vélib — Départ & Arrivée",
    page_icon="🚲",
    layout="wide",
)

# ========= Constantes =========
API_STATIONS = (
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
)

# ========= Classe Predictor (minimale) =========
class Predictor:
    """
    - load(path) / load_bytes(raw)
    - predict_proba(df), predict_label(df, threshold)
    Le pipeline doit contenir une étape 'preprocessor'.
    """

    def __init__(self, timezone: str = "Europe/Paris", target_col: str = "target_empty", verbose: bool = True):
        self.timezone = timezone
        self.target_col = target_col
        self.verbose = verbose
        self.pipeline_ = None
        self.preprocessor_ = None
        # ⚠️ Doit matcher EXACTEMENT le modèle entraîné
        self.feature_list_ = (
            ["hour_ssin", "hour_ccos", "dow_sin", "dow_cos", "month"]
            + ["holiday_flag", "is_weekend", "operative", "pluie", "vent", "soleil", "nuage"]
            + ["station_name"]
        )

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # On suppose 'time' en UTC
        ts = pd.to_datetime(df["time"], utc=True)
        out = df.copy()
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.weekday
        out["month"] = ts.dt.month
        out["hour_ssin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_ccos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7)
        out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7)
        return out

    def _build_preprocessor_if_needed(self):
        if self.preprocessor_ is None and self.pipeline_ is not None:
            if hasattr(self.pipeline_, "named_steps") and "preprocessor" in self.pipeline_.named_steps:
                self.preprocessor_ = self.pipeline_.named_steps["preprocessor"]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = self._add_time_features(df)
        self._build_preprocessor_if_needed()
        missing = [c for c in self.feature_list_ if c not in df2.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes pour la prédiction: {missing}")
        return df2[self.feature_list_]

    def load(self, path: str) -> None:
        self.pipeline_ = joblib.load(path)
        self._build_preprocessor_if_needed()

    def load_bytes(self, raw_bytes: bytes) -> None:
        if not raw_bytes:
            raise ValueError("Fichier modèle vide (0 octet).")
        try:
            self.pipeline_ = joblib.load(BytesIO(raw_bytes))
            self._build_preprocessor_if_needed()
        except EOFError:
            raise ValueError("Le fichier est corrompu ou n'est pas un modèle joblib valide.")
        except Exception as e:
            raise ValueError(f"Impossible de charger le modèle: {e}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self.pipeline_, "named_steps")
        X = self._prepare_features(df)
        return self.pipeline_.predict_proba(X)[:, 1]

    def predict_label(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        p = self.predict_proba(df)
        return pd.Series((p >= threshold).astype(int), index=df.index, name=f"{self.target_col}_pred")


# ========= Fonctions utilitaires =========
@st.cache_data(ttl=600)
def load_stations() -> pd.DataFrame:
    """Charge la table station_information depuis l'API officielle."""
    r = requests.get(API_STATIONS, timeout=20)
    r.raise_for_status()
    data = r.json()
    stations = data["data"]["stations"]
    df = pd.DataFrame(stations)
    # Colonnes utiles
    return df[["station_id", "name", "capacity", "lat", "lon"]]

def make_model_input_from_stations(
    df_st: pd.DataFrame,
    utc_time_iso: str,
    holiday_flag: bool,
    is_weekend: bool,
    operative: bool,
    pluie: int,
    vent: int,
    soleil: int,
    nuage: int,
) -> pd.DataFrame:
    """
    Construit le DataFrame d'entrée modèle pour les stations sélectionnées
    (schéma light: uniquement features temporelles + flags + station_name).
    """
    df = df_st.copy()
    df["time"] = utc_time_iso

    # Flags globaux
    df["holiday_flag"] = int(holiday_flag)
    df["is_weekend"] = int(is_weekend)
    df["operative"] = int(operative)
    df["pluie"] = int(pluie)
    df["vent"] = int(vent)
    df["soleil"] = int(soleil)
    df["nuage"] = int(nuage)

    # Le modèle attend 'station_name'
    df["station_name"] = df["name"].astype(str)

    # On garde des colonnes meta pour l'affichage
    df = df[
        [
            "time",
            "holiday_flag", "is_weekend", "operative",
            "pluie", "vent", "soleil", "nuage",
            "station_name",  # utilisé par le modèle
            "station_id", "name", "capacity", "lat", "lon",  # meta (affichage)
        ]
    ]
    return df

def normalize_schema_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise types/valeurs (évite soucis OHE).
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["station_name"] = out["station_name"].astype(str)
    for c in ["holiday_flag", "is_weekend", "operative", "pluie", "vent", "soleil", "nuage"]:
        out[c] = (pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int) != 0).astype(int)
    return out

def save_upload_to_tmp(uploaded_file) -> str:
    """Écrit l'upload dans un fichier temporaire (fallback)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ========= UI : Titre & Explications =========
st.title("🚲 Prédictions Vélib — Départ & Arrivée (toutes stations / stations choisies)")
st.caption(
    "Charge les modèles, choisis l'heure et les flags météo. "
    "Tu peux sélectionner **les stations** depuis un **CSV** (multiselect avec « Tout sélectionner »)."
)

# ========= Sidebar : Paramètres globaux =========
st.sidebar.header("⚙️ Paramètres globaux")

# Date / heure (Europe/Paris) → UTC
col_dt1, col_dt2 = st.sidebar.columns(2)
d: date = col_dt1.date_input("Date (Europe/Paris)", value=datetime.now().date())
t: time = col_dt2.time_input("Heure (Europe/Paris)", value=time(8, 0))

tz_paris = pytz.timezone("Europe/Paris")
local_dt = tz_paris.localize(datetime.combine(d, t))
utc_iso = local_dt.astimezone(pytz.utc).isoformat()
st.sidebar.markdown(f"**Heure UTC** utilisée par le modèle : `{utc_iso}`")

# Flags calendrier & météo
col_flags1, col_flags2, col_flags3 = st.sidebar.columns(3)
holiday_flag = col_flags1.checkbox("Jour férié ?", value=False)
is_weekend = col_flags2.checkbox("Week-end ?", value=(d.weekday() >= 5))
operative = col_flags3.checkbox("Station opérationnelle ?", value=True)

st.sidebar.divider()
st.sidebar.header("🌦️ Météo (flags simples)")
col_w1, col_w2, col_w3, col_w4 = st.sidebar.columns(4)
pluie = col_w1.checkbox("Pluie", value=False)
vent   = col_w2.checkbox("Vent", value=False)
soleil = col_w3.checkbox("Soleil", value=True)
nuage  = col_w4.checkbox("Nuage", value=False)

# ========= Chargement stations (API) =========
try:
    df_stations_api = load_stations()
except Exception as e:
    st.error(f"Erreur lors du chargement des stations: {e}")
    st.stop()

# ========= Sélection stations via CSV =========
st.sidebar.divider()
st.sidebar.header("📄 Stations depuis un CSV (optionnel)")
st.sidebar.caption("Le CSV doit contenir une colonne **station_name** ou **name**.")

csv_file = st.sidebar.file_uploader("Importer CSV stations", type=["csv"])
csv_station_names = None

if csv_file is not None:
    try:
        df_csv = pd.read_csv(csv_file)
        # Détecte le nom de colonne
        if "station_name" in df_csv.columns:
            csv_station_names = (
                df_csv["station_name"].astype(str).dropna().drop_duplicates().sort_values().tolist()
            )
        elif "name" in df_csv.columns:
            csv_station_names = (
                df_csv["name"].astype(str).dropna().drop_duplicates().sort_values().tolist()
            )
        else:
            st.warning("CSV importé, mais aucune colonne 'station_name' ni 'name' trouvée. On ignore le CSV.")
            csv_station_names = None
    except Exception as e:
        st.error(f"Impossible de lire le CSV: {e}")
        csv_station_names = None

# Si CSV disponible, multiselect basé sur CSV. Sinon, multiselect basé sur l'API.
if csv_station_names is not None:
    options_station_names = csv_station_names
    source_label = "CSV"
else:
    options_station_names = (
        df_stations_api["name"].astype(str).dropna().drop_duplicates().sort_values().tolist()
    )
    source_label = "API"

st.sidebar.subheader(f"🎯 Sélection des stations (source: {source_label})")
select_all = st.sidebar.checkbox("Tout sélectionner", value=True)
selected_names = st.sidebar.multiselect(
    "Stations à prédire",
    options=options_station_names,
    default=options_station_names if select_all else [],
    help="Tu peux filtrer/chercher puis cocher.",
)

# ========= Aperçu des stations (API) =========
with st.expander("📋 Aperçu des stations API (brut)", expanded=False):
    st.dataframe(df_stations_api.head(20), width="stretch")

# ========= Sous-ensemble à prédire =========
if selected_names:
    # Filtre les stations API sur la sélection (join sur 'name')
    df_stations = df_stations_api[df_stations_api["name"].astype(str).isin(selected_names)].copy()
else:
    # Si rien sélectionné → rien à prédire (on peut aussi choisir "toutes")
    df_stations = df_stations_api.iloc[0:0].copy()  # DataFrame vide

st.caption(f"Stations retenues: **{len(df_stations):,}**")

# ========= Construction DataFrame d'entrée =========
df_input_all = make_model_input_from_stations(
    df_stations, utc_iso, holiday_flag, is_weekend, operative,
    int(pluie), int(vent), int(soleil), int(nuage)
)

# ========= Modèles =========
st.sidebar.divider()
st.sidebar.header("📦 Modèles")
uploaded_depart = st.sidebar.file_uploader("Modèle départ (target_empty) .joblib", type=["joblib"])
uploaded_arrive = st.sidebar.file_uploader("Modèle arrivée (target_full) .joblib", type=["joblib"])

top_k = st.sidebar.slider("Afficher top K stations", 5, 100, 20, 1)
run = st.sidebar.button("🚀 Lancer la prédiction")

# ========= Prédiction =========
colA, colB = st.columns(2)

if run:
    if uploaded_depart is None or uploaded_arrive is None:
        st.error("Merci de déposer **les deux** modèles (.joblib) avant de lancer.")
        st.stop()
    if df_stations.empty:
        st.warning("Aucune station sélectionnée. Merci de choisir au moins une station.")
        st.stop()

    # Option 1: chargement direct en mémoire (recommandé)
    bytes_depart = uploaded_depart.getvalue()
    bytes_arrive = uploaded_arrive.getvalue()

    pred_depart = Predictor(target_col="target_empty", timezone="Europe/Paris")
    pred_arriv  = Predictor(target_col="target_full",  timezone="Europe/Paris")

    try:
        pred_depart.load_bytes(bytes_depart)
        pred_arriv.load_bytes(bytes_arrive)
    except ValueError as e:
        # Fallback vers fichiers temporaires si besoin
        st.warning(f"Chargement en mémoire impossible: {e}. Tentative via fichiers temporaires…")
        path_depart = save_upload_to_tmp(uploaded_depart)
        path_arrive = save_upload_to_tmp(uploaded_arrive)
        pred_depart.load(path_depart)
        pred_arriv.load(path_arrive)

    # Normalisation stricte du schéma avant prédire (inclut météo + station_name)
    df_for_pred = normalize_schema_for_model(df_input_all.copy())

    # Prédictions
    with st.spinner(f"Prédiction en cours sur {len(df_for_pred):,} stations..."):
        proba_depart = pred_depart.predict_proba(df_for_pred)
        label_depart = (proba_depart >= 0.5).astype(int)

        proba_arriv = pred_arriv.predict_proba(df_for_pred)
        label_arriv = (proba_arriv >= 0.5).astype(int)

    # Résultats combinés (on garde des méta pour l'affichage)
    df_res = df_for_pred[["station_name", "station_id", "name", "capacity", "lat", "lon"]].copy()
    df_res["proba_velo_depart"] = proba_depart
    df_res["velo_dispo_depart"] = label_depart
    df_res["proba_place_arrivee"] = proba_arriv
    df_res["place_dispo_arrivee"] = label_arriv

    # Affichages
    with colA:
        st.subheader("🟢 Top stations — vélos dispo (départ)")
        df_top_bikes = (
            df_res[df_res["velo_dispo_depart"] == 1]
            .sort_values("proba_velo_depart", ascending=False)
            .head(top_k)
        )
        st.dataframe(
            df_top_bikes[["station_name", "proba_velo_depart", "capacity", "station_id"]],
            width="stretch",
        )
        st.caption("Seuil 0.5 (modifiable dans le code si besoin).")

    with colB:
        st.subheader("🟦 Top stations — places dispo (arrivée)")
        df_top_docks = (
            df_res[df_res["place_dispo_arrivee"] == 1]
            .sort_values("proba_place_arrivee", ascending=False)
            .head(top_k)
        )
        st.dataframe(
            df_top_docks[["station_name", "proba_place_arrivee", "capacity", "station_id"]],
            width="stretch",
        )

    st.divider()
    st.subheader("📥 Export")
    csv = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les résultats (CSV)",
        data=csv,
        file_name="predictions_velib.csv",
        mime="text/csv",
        width="stretch",
    )

else:
    st.info(
        "☝️ Charge tes deux modèles (.joblib), importe un CSV de stations (optionnel), "
        "sélectionne les stations dans la liste (ou coche « Tout sélectionner »), puis clique **Lancer la prédiction**."
    )
