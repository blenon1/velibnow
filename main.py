from core.database import DatabaseManager
from etl.velib_etl import VelibETL
from etl.weather_etl import WeatherETL
from etl.holidays_etl import HolidaysETL

if __name__ == "__main__":
    db = DatabaseManager()

    print("➡️ ETL Vélib")
    VelibETL(db).run()

    print("➡️ ETL Météo")
    WeatherETL(db).run()

    print("➡️ ETL Jours Fériés / Vacances")
    HolidaysETL(db).run()
