from etl.velib_etl import VelibETL
from etl.weather_etl import WeatherETL
from etl.holidays_etl import HolidaysETL
from features.builder import FeatureBuilder
from modeling.model_trainer import ModelTrainer

if __name__ == "__main__":
    print("➡️ ETL Vélib")
    velib_data = VelibETL().run()

    print("➡️ ETL Météo")
    WeatherETL().run()

    print("➡️ ETL Jours Fériés / Vacances")
    HolidaysETL().run()

    print("➡️ Création des features")
    df = FeatureBuilder().build()

    print("➡️ Entraînement du modèle")
    ModelTrainer().train(df)

    print("✅ Pipeline terminé avec succès !")
