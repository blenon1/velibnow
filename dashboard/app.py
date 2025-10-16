# dashboard/app.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np              # pour le scaling du rayon
import pydeck as pdk            # pour la carte
import pytz
from datetime import datetime, date, time

from data_api import (
    get_stations_from_backend,
    get_station_names_from_csv_backend,
    predict_remote_backend,
)

st.set_page_config(page_title="Prédictions Vélib — API-driven", page_icon="🚲", layout="wide")

# ---- Paramètres globaux ----
st.sidebar.header("⚙️ Paramètres globaux")
col_dt1, col_dt2 = st.sidebar.columns(2)
d: date = col_dt1.date_input("Date (Europe/Paris)", value=datetime.now().date())
t: time = col_dt2.time_input("Heure (Europe/Paris)", value=time(8, 0))

tz_paris = pytz.timezone("Europe/Paris")
utc_iso = tz_paris.localize(datetime.combine(d, t)).astimezone(pytz.utc).isoformat()
st.sidebar.caption(f"Heure UTC utilisée : `{utc_iso}`")

col_flags1, col_flags2, col_flags3 = st.sidebar.columns(3)
holiday_flag = col_flags1.checkbox("Jour férié ?", value=False)
is_weekend = col_flags2.checkbox("Week-end ?", value=(d.weekday() >= 5))
operative = col_flags3.checkbox("Station opérationnelle ?", value=True)

st.sidebar.header("🌦️ Météo (flags)")
col_w1, col_w2, col_w3, col_w4 = st.sidebar.columns(4)
pluie = col_w1.checkbox("Pluie", value=False)
vent = col_w2.checkbox("Vent", value=False)
soleil = col_w3.checkbox("Soleil", value=True)
nuage = col_w4.checkbox("Nuage", value=False)

run = st.sidebar.button("🚀 Lancer la prédiction")

# ---- Haut de page : info API + info CSV (backend) ----
st.title("🚲 Prédictions Vélib — Back piloté (modèles + CSV côté serveur)")
st.caption("Le tableau API est purement informatif. La prédiction utilise la **liste station_name du CSV côté backend**.")

with st.expander("📋 Aperçu des stations (API Vélib)", expanded=False):
    try:
        df_api = get_stations_from_backend()
        st.dataframe(df_api.head(20), width="stretch")
        st.caption(f"Total (API): {len(df_api):,}")
    except Exception as e:
        st.error(f"Erreur API backend /stations : {e}")

with st.expander("📄 Liste des station_name (CSV côté backend)", expanded=True):
    try:
        names = get_station_names_from_csv_backend()
        st.write(f"Total dans CSV: **{len(names):,}**")
        st.dataframe(pd.DataFrame({"station_name": names}).head(30), width="stretch")
    except Exception as e:
        st.error(f"Erreur backend /stations/csv : {e}")

# ---- Lancer prédiction (tout est fait côté backend) ----
if run:
    with st.spinner("Prédiction côté backend..."):
        try:
            df_res = predict_remote_backend(
                utc_time_iso=utc_iso,
                holiday_flag=holiday_flag,
                is_weekend=is_weekend,
                operative=operative,
                pluie=int(pluie),
                vent=int(vent),
                soleil=int(soleil),
                nuage=int(nuage),
            )
        except Exception as e:
            st.error(f"Erreur backend /predict : {e}")
            st.stop()

    st.subheader("🟢 Résultats (triés départ)")
    if df_res.empty:
        st.warning("Aucun résultat.")
    else:
        st.dataframe(
            df_res.sort_values("proba_velo_depart", ascending=False),
            width="stretch",
        )

        # --- 🗺️ Carte des stations ---
        st.subheader("🗺️ Carte des stations prédites")

        if not {"lat", "lon"}.issubset(df_res.columns):
            st.info("Pas de colonnes `lat`/`lon` dans le résultat — impossible d’afficher la carte.")
        else:
            proba_col = st.radio(
                "Probabilité à afficher en intensité/couleur :",
                ["proba_velo_depart", "proba_place_arrivee"],
                horizontal=True,
            )

            map_df = df_res.copy()
            map_df = map_df.dropna(subset=["lat", "lon"])
            map_df = map_df[(map_df["lat"].between(-90, 90)) & (map_df["lon"].between(-180, 180))]

            if map_df.empty:
                st.info("Aucune coordonnée valide pour la carte.")
            else:
                # Radius (mètres) selon proba: 50m à 300m
                p = map_df[proba_col].clip(0, 1).fillna(0)
                map_df["radius"] = (50 + 250 * p).astype(float)

                # Champs texte formatés pour le tooltip (pydeck ne gère pas {:.2f})
                map_df["proba_depart_txt"]  = map_df["proba_velo_depart"].fillna(0).map(lambda x: f"{x:.2f}")
                map_df["proba_arrivee_txt"] = map_df["proba_place_arrivee"].fillna(0).map(lambda x: f"{x:.2f}")

                # Couleur: rouge → vert selon la proba sélectionnée
                map_df["r"] = (255 * (1 - p)).astype(int)
                map_df["g"] = (255 * p).astype(int)
                map_df["b"] = 60
                map_df["a"] = 160

                view = pdk.ViewState(
                    latitude=map_df["lat"].mean(),
                    longitude=map_df["lon"].mean(),
                    zoom=11,
                    pitch=0,
                    bearing=0,
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="[r, g, b, a]",
                    pickable=True,
                    auto_highlight=True,
                )

                tooltip = {
                    "html": (
                        "<b>{name}</b><br/>"
                        "ID: {station_id}<br/>"
                        "Proba départ: {proba_depart_txt}<br/>"
                        "Proba arrivée: {proba_arrivee_txt}"
                    ),
                    "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
                }

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view,
                        tooltip=tooltip,
                        map_style="light",
                    ),
                    use_container_width=True,
                )
        # --- fin carte ---

        st.divider()
        csv = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", data=csv, file_name="predictions_velib.csv", mime="text/csv", width="stretch")
else:
    st.info("Régle les paramètres, puis lance la prédiction : le backend charge les modèles + CSV et renvoie les résultats.")
