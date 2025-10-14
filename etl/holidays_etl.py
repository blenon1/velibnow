import pandas as pd
from data.holidays_api import HolidaysAPI
from etl.base_etl import BaseETL

class HolidaysETL(BaseETL):
    def __init__(self, db):
        super().__init__(db, "calendar_events")
        self.api = HolidaysAPI()

    def extract(self):
        print("⬇️ Extraction jours fériés et vacances...")
        holidays = self.api.fetch_public_holidays()
        vacations = self.api.fetch_school_vacations()
        holidays["type"] = "holiday"
        vacations["type"] = "vacation"
        return pd.concat([holidays, vacations], ignore_index=True)

    def transform(self, df):
        print("🧹 Transformation calendrier...")
        df["start_date"] = pd.to_datetime(df.get("start_date", None))
        df["end_date"] = pd.to_datetime(df.get("end_date", None))
        return df
