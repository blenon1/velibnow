from data.velib_api import VelibAPI
from core.database import DatabaseManager
from config import POSTGRES_URL

class VelibETL:
    def __init__(self):
        self.api = VelibAPI()
        self.db = DatabaseManager(POSTGRES_URL)

    def run(self):
        df = self.api.fetch_data()
        self.db.save("velib_status", df)
        return df
