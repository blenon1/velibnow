import pandas as pd
from config import HISTORICAL_VELLIB_CSV_PATH

class VelibCsvReader:
    def __init__(self):
        self.file_path = HISTORICAL_VELLIB_CSV_PATH

    def read_dataframe(self):
        """Combine status + info sur les stations"""
        dataframe = pd.read_csv(
            filepath_or_buffer=self.file_path,
            sep=","
        )
        return dataframe