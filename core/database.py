from sqlalchemy import create_engine
import pandas as pd

class DatabaseManager:
    def __init__(self, url):
        self.engine = create_engine(url)

    def save(self, table_name, df: pd.DataFrame):
        df.to_sql(table_name, self.engine, if_exists="append", index=False)

    def read(self, query: str):
        return pd.read_sql(query, self.engine)
