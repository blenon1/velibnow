import requests, pandas as pd
from config import URL_HOLIDAYS, URL_SCHOOL

class HolidaysAPI:
    def fetch_public_holidays(self) -> pd.DataFrame:
        data = requests.get(URL_HOLIDAYS).json()
        return pd.DataFrame(list(data.items()), columns=["date", "holiday_name"])

    def fetch_school_vacations(self) -> pd.DataFrame:
        data = requests.get(URL_SCHOOL).json()["results"]
        df = pd.DataFrame(data)
        return df[["start_date", "end_date", "zones", "description"]]
