import requests, pandas as pd
from config import URL_METEO_HOURLY

class WeatherAPI:
    def fetch_hourly(self) -> pd.DataFrame:
        data = requests.get(URL_METEO_HOURLY).json()
        df = pd.DataFrame({
            "datetime": data["hourly"]["time"],
            "temperature_2m": data["hourly"]["temperature_2m"]
        })
        return df
