from data.weather_api import WeatherAPI
from core.database import DatabaseManager
from config import POSTGRES_URL

class WeatherETL:
    def __init__(self):
        self.api = WeatherAPI()
        self.db = DatabaseManager(POSTGRES_URL)

    def run(self):
        df = self.api.fetch_hourly()
        self.db.save("weather_hourly", df)
        return df
