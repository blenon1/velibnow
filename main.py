from __future__ import annotations

import pandas as pd

from core.database import DatabaseManager
from data import velib_historical_csv, weather_historical_csv, holidays_api
from features.feature_builder import FeatureBuilder
from modeling.model_trainer import VelibSimpleModel 

if __name__ == "__main__":
    db = DatabaseManager()
    
    def read_pg_table(table_name: str):
        print(f"➡️ Start reading pg table {table_name}")
        dataframe = pd.read_sql(f"SELECT * FROM {table_name}", db.engine)
        print(f"✅ pg table {table_name} successfuly read")
        return dataframe
    

    # ===== VELIB =====
    
    if input("Would you run VELIB STEP (Yes/No) ? ") == "Yes":
        print("➡️ Start load historical Velib data and saving to postgresql")
        db.save(
            table_name="velib",
            df= (
                velib_historical_csv.
                VelibCsvReader().
                read_dataframe()
            )
        )
        print("✅ Velib data saved successfully !")

    # ===== WEATHER =====

    if input("Would you run WEATHER STEP (Yes/No) ? ") == "Yes":
        print("➡️ Start load historical Weather data and saving to postgresql")
        db.save(
            table_name="weathers",
            df=(
                weather_historical_csv
                .WeatherCsvReader()
                .read_standardized()
            )
        )
        print("✅ Weather data saved successfully !")

    # ===== HOLIDAYS =====

    if input("Would you run HOLIDAYS STEP (Yes/No) ? ") == "Yes":    
        print("➡️ Start load Holidays data")
        holidays = holidays_api.HolidaysAPI()

        public_df   = holidays.fetch_public_holidays()
        vacations_df = holidays.fetch_school_vacations()

        public_df["type"]    = "holiday"
        vacations_df["type"] = "vacation"

        print("➡️ Start concat dataframe and saving to postgresql")
        db.save(
            table_name="calendar",
            df=(
                pd.concat(
                    [public_df, vacations_df],
                    ignore_index=True
                )
            )
        )
        print("✅ Holidays data saved successfully !")

    # ===== CREATE FEATURES AND JOINS =====

    if input("Would you run FEATURES STEP (Yes/No) ? ") == "Yes":       
        print("➡️ Start loading needed table from postgresql and generating feature and joining")
        
        db.save(
            table_name="features",
            df=FeatureBuilder(
                velib=read_pg_table("velib"),
                weather=read_pg_table("weathers"),
                calendar=read_pg_table("calendar")
            ).run()
        )
        
        print("✅ Features data saved successfully !")
    
    if input("Would you train a model (Yes/No) ? ") == "Yes":   
        model = input("whitch model (target_full/target_empty) ?")
        if model in ("target_full", "target_empty"):
            
            feature_dataframe = read_pg_table("features")
            # --- 2. Config du modèle ---
            cfg = {
                "target_col": model,   # "target_empty" ou "target_full"
                "model_type": "gb",         # "gb" ou "logit"
                "timezone": "Europe/Paris",
                "random_state": 42,
                "test_size": 0.2,
                "verbose": True,
            }
            model = VelibSimpleModel.from_config(cfg)

            # --- 3. Entraînement ---
            metrics = model.fit(feature_dataframe)
            print("📊 Résultats:")
            for k, v in metrics.items():
                print(f"  - {k}: {v}")

            # --- 4. Sauvegarde ---
            model_type = "full" if model == "target_full" else "empty"
            model_path = f"./modeling/models_storage/model_{model_type}.joblib"
            model.save(model_path)
            print(f"💾 Modèle sauvegardé → {model_path}")
            
        else : print("!!! Bad model selection !!!")
