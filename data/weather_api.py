import requests
import pandas as pd
from config import URL_METEO_HOURLY

class WeatherAPI:
    """
    Récupération des températures horaires via Open-Meteo.
    """

    def fetch_hourly(self) -> pd.DataFrame:
        try:
            data = requests.get(URL_METEO_HOURLY, timeout=10).json()
            hourly = data.get("hourly", {})
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly.get("time", [])),
                "temperature_2m": hourly.get("temperature_2m", [])
            })
            return df
        except Exception as e:
            raise RuntimeError(f"Erreur API météo : {e}")
