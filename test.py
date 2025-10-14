from sqlalchemy import create_engine
from config import POSTGRES_URL

print("🔍 Test de connexion à PostgreSQL...")
engine = create_engine(POSTGRES_URL)
with engine.connect() as conn:
    print("✅ Connexion réussie !")
