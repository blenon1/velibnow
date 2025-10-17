import pandas as pd
import numpy as np

class FeatureBuilder:
    def __init__(self, velib, weather, calendar):
        self.velib = velib
        self.weather = weather
        self.calendar = calendar

    def _preprocess(self, velib, weather, calendar):
        """Prépare et fusionne les datasets (fusion horaire simple)"""
        print("🧹 Préparation des données...")

        # Nettoyage des dates (UTC)
        weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce", utc=True)
        velib["time"] = pd.to_datetime(velib["time"], errors="coerce", utc=True)

        # Clé horaire
        weather["ts_hour"] = weather["timestamp"].dt.floor("h")
        velib["ts_hour"] = velib["time"].dt.floor("h")

        # ===== Flags météo très simples (0/1) =====
        precip = pd.to_numeric(weather.get("precip_mm", 0), errors="coerce").fillna(0.0)
        precip_dur = pd.to_numeric(weather.get("precip_dur_min", 0), errors="coerce").fillna(0.0)
        ws = pd.to_numeric(weather.get("wind_speed_10m_ms", 0), errors="coerce").fillna(0.0)
        wg = pd.to_numeric(weather.get("wind_gust_10m_ms", 0), errors="coerce").fillna(0.0)
        cloud_oktas = pd.to_numeric(weather.get("cloud_oktas", np.nan), errors="coerce")

        # Règles binaires
        weather["pluie"]  = ((precip >= 0.1) | (precip_dur >= 5)).astype(int)
        weather["vent"]   = ((ws >= 8.0) | (wg >= 10.8)).astype(int)
        weather["soleil"] = cloud_oktas.le(2).fillna(False).astype(int)
        weather["nuage"]  = cloud_oktas.ge(6).fillna(False).astype(int)

        # 1 ligne par heure
        weather_flags = (
            weather.sort_values("timestamp")
                   .drop_duplicates(subset=["ts_hour"], keep="last")
                   [["ts_hour", "pluie", "vent", "soleil", "nuage"]]
                   .copy()
        )
        weather_flags[["pluie", "vent", "soleil", "nuage"]] = weather_flags[["pluie", "vent", "soleil", "nuage"]].fillna(0).astype(int)

        # Fusion météo ↔ Vélib
        df = velib.merge(weather_flags, on="ts_hour", how="left")
        for c in ["pluie", "vent", "soleil", "nuage"]:
            if c not in df.columns:
                df[c] = 0
        df[["pluie", "vent", "soleil", "nuage"]] = df[["pluie", "vent", "soleil", "nuage"]].fillna(0).astype(int)

        # Ajout calendrier
        df["date"] = df["time"].dt.tz_convert("Europe/Paris").dt.date
        cal = calendar.copy()
        cal["date"] = pd.to_datetime(cal["start_date"]).dt.date
        cal["holiday_flag"] = 1
        df = df.merge(cal[["date", "holiday_flag"]].drop_duplicates(), on="date", how="left")
        df["holiday_flag"] = df["holiday_flag"].fillna(0).astype(int)

        return df

    def _feature_engineering(self, df):
        """Crée des variables dérivées utiles pour la prédiction"""
        print("⚙️ Construction des features...")

        # TOTAL vélos dispo & bornes libres
        df["available_total"] = df["available_mechanical"].fillna(0) + df["available_electrical"].fillna(0)
        df["docks_available"] = df["capacity"].fillna(0) - df["available_total"]

        # Taux d'occupation
        df["fill_rate"] = (df["available_total"] / df["capacity"].replace(0, pd.NA)).fillna(0).clip(0, 1)

        # ✅ Calcul des ratios sûrs
        ratio_empty = (df["docks_available"] / df["capacity"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
        ratio_full = (df["available_total"] / df["capacity"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

        # ✅ Définition automatique des seuils (adaptés à ta distribution)
        empty_threshold = ratio_empty.quantile(0.70)  # stations avec beaucoup de place
        full_threshold  = ratio_full.quantile(0.30)   # stations avec assez de vélos

        print(f"📏 Seuils appliqués: target_empty >= {empty_threshold:.2f} | target_full >= {full_threshold:.2f}")

        # ✅ Cibles binaires équilibrées
        df["target_empty"] = (ratio_empty >= empty_threshold).astype(int)
        df["target_full"]  = (ratio_full >= full_threshold).astype(int)

        # Heures / jours
        df = df.sort_values(["station_name", "time"])
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.day_name()

        # Moyenne glissante (3h) par station
        df["rolling_fill_rate"] = (
            df.groupby("station_name")["fill_rate"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

        # Week-end
        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

        return df

    def run(self):
        """Exécution complète du pipeline de feature engineering"""
        merged = self._preprocess(self.velib, self.weather, self.calendar)
        if merged is None:
            raise RuntimeError("[FeatureBuilder.run] _preprocess a renvoyé None (attendu: DataFrame).")
        features = self._feature_engineering(merged)
        return features