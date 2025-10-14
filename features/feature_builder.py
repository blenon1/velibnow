import pandas as pd
from core.database import DatabaseManager


class FeatureBuilder:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def load_data(self):
        """Charge les trois tables de la base PostgreSQL"""
        velib = pd.read_sql("SELECT * FROM velib_status", self.db.engine)
        weather = pd.read_sql("SELECT * FROM weather_data", self.db.engine)
        calendar = pd.read_sql("SELECT * FROM calendar_events", self.db.engine)

        print(f"📦 Chargé : {len(velib)} Vélib, {len(weather)} météo, {len(calendar)} événements")
        return velib, weather, calendar

    def preprocess(self, velib, weather, calendar):
        """Prépare et fusionne les datasets"""
        print("🧹 Préparation des données...")

        # Nettoyage des dates
        weather["datetime"] = pd.to_datetime(weather["datetime"])
        velib["last_reported"] = pd.to_datetime(velib["last_reported"], errors="coerce")

        # Fusion sur le jour
        weather["date"] = weather["datetime"].dt.date
        velib["date"] = velib["last_reported"].dt.date

        # Fusion météo ↔ Vélib
        df = pd.merge(velib, weather[["date", "temperature_2m"]], on="date", how="left")

        # Ajout indicateurs calendrier
        calendar["date"] = pd.to_datetime(calendar["start_date"]).dt.date
        calendar["holiday_flag"] = 1
        df = pd.merge(df, calendar[["date", "holiday_flag"]].drop_duplicates(), on="date", how="left")
        df["holiday_flag"] = df["holiday_flag"].fillna(0)

        return df

    def feature_engineering(self, df):
        """Crée des variables dérivées utiles pour la prédiction"""
        print("⚙️ Construction des features...")

        # Ratio de disponibilité
        df["fill_rate"] = df["num_bikes_available"] / (df["num_bikes_available"] + df["num_docks_available"])
        df["fill_rate"] = df["fill_rate"].fillna(0).clip(0, 1)

        # Heures et jours
        df["hour"] = df["last_reported"].dt.hour
        df["day_of_week"] = df["last_reported"].dt.day_name()

        # Moyenne glissante par station (3 dernières heures)
        df = df.sort_values(["station_id", "last_reported"])
        df["rolling_fill_rate"] = (
            df.groupby("station_id")["fill_rate"]
              .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

        # Température normalisée
        df["temp_norm"] = (df["temperature_2m"] - df["temperature_2m"].mean()) / df["temperature_2m"].std()

        # Indicateur jour/week-end
        df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

        return df

    def save(self, df):
        """Sauvegarde le dataset enrichi dans PostgreSQL"""
        table_name = "features_dataset"
        self.db.save(table_name, df)
        print(f"✅ Features sauvegardées dans {table_name} ({len(df)} lignes)")

    def run(self):
        """Exécution complète du pipeline de feature engineering"""
        velib, weather, calendar = self.load_data()
        merged = self.preprocess(velib, weather, calendar)
        features = self.feature_engineering(merged)
        self.save(features)
        return features


if __name__ == "__main__":
    from config import POSTGRES_URL
    db = DatabaseManager()
    builder = FeatureBuilder(db)
    builder.run()
