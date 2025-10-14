from abc import ABC, abstractmethod
from core.database import DatabaseManager

class BaseETL(ABC):
    def __init__(self, db: DatabaseManager, table_name: str):
        self.db = db
        self.table_name = table_name

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def load(self, df):
        print(f"💾 Insertion dans {self.table_name} ({len(df)} lignes)")
        self.db.save(self.table_name, df)

    def run(self):
        data = self.extract()
        df = self.transform(data)
        self.load(df)
        print(f"✅ ETL {self.table_name} terminé\n")
        return df
