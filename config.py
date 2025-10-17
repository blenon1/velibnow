POSTGRES_URL = "postgresql+psycopg2://myuser:mypassword@localhost:5432/velibnow"

URL_VELIB = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"
URL_HOLIDAYS = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"
URL_SCHOOL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
URL_METEO_HOURLY = "https://api.open-meteo.com/v1/forecast?latitude=48.8566&longitude=2.3522&hourly=temperature_2m"

MODEL_PATH = "model.pkl"
HORIZON_MIN = 30

HISTORICAL_VELLIB_CSV_PATH = "./data/dataset/historique_stations.csv"
WEATHER_HISTORICAL_CSV_PATH = "./data/dataset/historical_meteo.csv"
