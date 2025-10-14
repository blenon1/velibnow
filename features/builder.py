import pandas as pd
from core.database import DatabaseManager
from config import POSTGRES_URL

class FeatureBuilder:
    def __init__(self):
        self.db = DatabaseManager(POSTGRES_URL)

    def build(self):
        velib = self.db.read("SELECT * FROM velib_status")
        weather = self.db.read("SELECT * FROM weather_hourly")
        holidays = self.db.read("SELECT * FROM holidays")

        df = velib.merge(weather, left_on="duedate", right_on="datetime", how="left")
        df["is_holiday"] = df["duedate"].str[:10].isin(holidays["date"])
        df["ratio_occ"] = df["numbikesavailable"] / df["capacity"]
        return df
