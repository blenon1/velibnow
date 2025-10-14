from core.database import DatabaseManager
from etl.velib_etl import VelibETL
from etl.weather_etl import WeatherETL
from etl.holidays_etl import HolidaysETL
from features.feature_builder import FeatureBuilder

if __name__ == "__main__":
    db = DatabaseManager()

    # print("➡️ ETL Vélib")
    # VelibETL(db).run()

    # print("➡️ ETL Météo")
    # WeatherETL(db).run()

    # print("➡️ ETL Jours Fériés / Vacances")
    # HolidaysETL(db).run()
    
    print("\n🚀 Construction du dataset de features")
    builder = FeatureBuilder(db)
    features_df = builder.run()

    print("\n✅ Pipeline complet terminé !")
    print(f"   → Dataset enrichi sauvegardé : {len(features_df)} lignes")
