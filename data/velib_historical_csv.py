import pandas as pd

class VelibCsvReader:
    def __init__(self):
        self.file_path = "./data/dataset/historique_stations.csv"

    def read_dataframe(self):
        """Combine status + info sur les stations"""
        dataframe = pd.read_csv(
            filepath_or_buffer=self.file_path,
            sep=",",
            header=True
        )
        return dataframe
