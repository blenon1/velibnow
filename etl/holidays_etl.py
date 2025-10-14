from data.holidays_api import HolidaysAPI
from core.database import DatabaseManager
from config import POSTGRES_URL

class HolidaysETL:
    def __init__(self):
        self.api = HolidaysAPI()
        self.db = DatabaseManager(POSTGRES_URL)

    def run(self):
        holidays = self.api.fetch_public_holidays()
        self.db.save("holidays", holidays)

        school = self.api.fetch_school_vacations()
        self.db.save("school_vacations", school)
