from sqlalchemy import create_engine
import pandas as pd
from config import POSTGRES_URL


class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(POSTGRES_URL)

    def save(self, table_name: str, df: pd.DataFrame):
        """Sauvegarde un DataFrame dans PostgreSQL"""
        if df is not None and not df.empty:
            # Force tous les types en objet (utile pour les conversions JSON/text)
            df = df.astype(object)

            # Écrit dans la base
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)
            print(f"✅ Données insérées dans {table_name} ({len(df)} lignes)")
        else:
            print(f"⚠️ Aucune donnée à insérer dans {table_name}")
