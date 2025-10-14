import requests, pandas as pd
from config import URL_VELIB

class VelibAPI:
    def fetch_data(self) -> pd.DataFrame:
        data = requests.get(URL_VELIB).json()
        results = data["results"]
        df = pd.DataFrame([r for r in results])
        df = df.rename(columns={"stationcode": "station_id"})
        return df[["station_id", "name", "coordonnees_geo.lat", "coordonnees_geo.lon",
                   "capacity", "numdocksavailable", "numbikesavailable", "duedate"]]
