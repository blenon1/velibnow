import pandas as pd
from data.weather_api import WeatherAPI
from etl.base_etl import BaseETL


class WeatherETL(BaseETL):
    def __init__(self, db):
        super().__init__(db, "weather_data")
        self.api = WeatherAPI()

    def extract(self):
        print("⬇️ Extraction Météo...")
        return self.api.fetch_hourly()

    def transform(self, df):
        print("🧹 Transformation Météo...")

        # Conversion du champ datetime
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Nettoyage basique : suppression des valeurs manquantes
        df = df.dropna(subset=["datetime", "temperature_2m"])

        # Ajout d'une colonne de jour pour les analyses futures
        df["day"] = df["datetime"].dt.date

        return df
