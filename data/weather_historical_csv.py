import pandas as pd
from typing import Dict
from config import WEATHER_HISTORICAL_CSV_PATH

class WeatherCsvReader:
    def __init__(self, file_path: str = WEATHER_HISTORICAL_CSV_PATH):
        self.file_path = file_path
        
        # Mapping colonnes brutes
        self.WEATHER_COL_MAP = {
            "AAAAMMJJHH": "timestamp",
            "NUM_POSTE":  "station_id",
            "NOM_USUEL":  "station_name",
            "LAT":        "lat_deg",
            "LON":        "lon_deg",
            "ALTI":       "alt_m",
            "RR1":        "precip_mm",
            "DRR1":       "precip_dur_min",
            "T":          "temp_c",
            "TD":         "dewpoint_c",
            "U":          "humidity_rel_pct",
            "FF":         "wind_speed_10m_ms",
            "DD":         "wind_dir_deg",
            "FXI":        "wind_gust_10m_ms",
            "PSTAT":      "pressure_hpa_station",
            "PMER":       "pressure_hpa_sea",

            "N":          "cloud_oktas",
            "INS":        "insolation_min",           
            "GLO":        "global_radiation_j_cm2",   
            "WW":         "wmo_present_weather",      
        }

        # Schéma pour parser les données
        self.WEATHER_DTYPES: Dict[str, str] = {
            "station_id":          "string",
            "station_name":        "string",
            "lat_deg":             "float64",
            "lon_deg":             "float64",
            "alt_m":               "Int64",
            "timestamp":           "datetime64[ns]",
            "precip_mm":           "float64",
            "precip_dur_min":      "Int64",
            "temp_c":              "float64",
            "dewpoint_c":          "float64",
            "humidity_rel_pct":    "float64",
            "wind_speed_10m_ms":   "float64",
            "wind_dir_deg":        "float64",
            "wind_gust_10m_ms":    "float64",
            "pressure_hpa":        "float64",
            
            "cloud_oktas":   "float64",
            "insolation_min":        "float64",
            "global_radiation_j_cm2":    "float64",
            "wmo_present_weather":        "float64",
        }

    def read_dataframe(self) -> pd.DataFrame:
        """Lecture brute du CSV."""
        try:
            return pd.read_csv(self.file_path, sep=";")
        except Exception as e:
            raise RuntimeError(f"Erreur API météo : {e}")

    def _select_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garde uniquement les colonnes connues puis les renomme."""
        cols_src = df.columns.intersection(self.WEATHER_COL_MAP.keys())
        return df.loc[:, cols_src].rename(columns=self.WEATHER_COL_MAP)

    def read_standardized(self) -> pd.DataFrame:
        """
        Pipeline vectorisé :
        - select/rename
        - parse timestamp (AAAAMMJJHH)
        - coalesce pression (PSTAT prioritaire sur PMER)
        - cast global selon schéma
        - ordre de colonnes propre
        """
        df_raw = self.read_dataframe()
        df = self._select_rename(df_raw)

        # Parse timestamp (format AAAAMMJJHH)
        df = df.assign(
            timestamp=pd.to_datetime(
                df["timestamp"].astype("string"),
                format="%Y%m%d%H",
                errors="coerce"
            )
        )

        # Coalesce pression (priorité station -> mer) si colonnes présentes
        # (si absentes, bfill s'applique sur colonnes manquantes sans boucle)
        pressure_sources = df.filter(items=["pressure_hpa_station", "pressure_hpa_sea"])
        if not pressure_sources.empty:
            df["pressure_hpa"] = pressure_sources.bfill(axis=1).iloc[:, 0]

        # Ordre + cast en une seule passe (les colonnes manquantes seront ajoutées vides)
        df = (
            df
            .reindex(columns=self.WEATHER_DTYPES.keys()) 
            .astype(self.WEATHER_DTYPES, errors="ignore")
        )

        return df
