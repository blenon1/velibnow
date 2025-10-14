import pandas as pd
import json
from data.velib_api import VelibAPI
from etl.base_etl import BaseETL

class VelibETL(BaseETL):
    def __init__(self, db):
        super().__init__(db, "velib_status")
        self.api = VelibAPI()

    def extract(self):
        print("⬇️ Extraction Vélib...")
        return self.api.fetch_combined()

    def transform(self, df):
        print("🧹 Transformation Vélib...")

        # Convertir les objets non scalaires (dict, list) en JSON string
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )

        # Nettoyage des dates
        if "last_reported" in df.columns:
            df["last_reported"] = pd.to_datetime(df["last_reported"], errors="coerce")

        # Ajout du taux de remplissage
        if "num_bikes_available" in df.columns and "capacity" in df.columns:
            df["fill_rate"] = (
                df["num_bikes_available"].astype(float) / df["capacity"].replace(0, 1)
            ).round(2)

        return df
