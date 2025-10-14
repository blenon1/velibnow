import requests
import pandas as pd

class VelibAPI:
    def __init__(self):
        self.status_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
        self.info_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"

    def fetch_status(self):
        data = requests.get(self.status_url).json()
        return pd.DataFrame(data["data"]["stations"])

    def fetch_info(self):
        data = requests.get(self.info_url).json()
        return pd.DataFrame(data["data"]["stations"])

    def fetch_combined(self):
        """Combine status + info sur les stations"""
        status_df = self.fetch_status()
        info_df = self.fetch_info()

        # Jointure sur station_id
        merged = pd.merge(info_df, status_df, on="station_id", suffixes=("_info", "_status"))
        return merged
