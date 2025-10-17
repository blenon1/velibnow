from sqlalchemy import create_engine
import pandas as pd
from config import POSTGRES_URL

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(POSTGRES_URL)

    def save(self, table_name: str, df: pd.DataFrame):
        """Sauvegarde un DataFrame dans PostgreSQL avec gestion mémoire simple."""
        if df is not None and not df.empty:
            print("✅ Données non vide -> écriture en cours")
            df = df.astype(object)
            
            chunk_size = 500_000
            df.to_sql(
                table_name,
                self.engine,
                if_exists="replace",
                index=False,
                chunksize=chunk_size
            )
            print(f"✅ Données insérées dans {table_name} ({len(df)} lignes, chunksize={chunk_size})")
        else:
            print(f"⚠️ Aucune donnée à insérer dans {table_name}")
