import requests
import pandas as pd
from config import URL_HOLIDAYS, URL_SCHOOL

class HolidaysAPI:
    """
    Récupération des jours fériés et des vacances scolaires françaises.
    """

    def fetch_public_holidays(self) -> pd.DataFrame:
        try:
            data = requests.get(URL_HOLIDAYS, timeout=10).json()
            df = pd.DataFrame(list(data.items()), columns=["date", "holiday_name"])
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            raise RuntimeError(f"Erreur API jours fériés : {e}")

    def fetch_school_vacations(self) -> pd.DataFrame:
        try:
            data = requests.get(URL_SCHOOL, timeout=10).json()
            results = data.get("results", [])
            df = pd.DataFrame(results)
            if not df.empty:
                df = df[["start_date", "end_date", "zones", "description"]]
                df["start_date"] = pd.to_datetime(df["start_date"])
                df["end_date"] = pd.to_datetime(df["end_date"])
            return df
        except Exception as e:
            raise RuntimeError(f"Erreur API vacances scolaires : {e}")
