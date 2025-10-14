import logging
from datetime import datetime
from sqlalchemy import create_engine

class Logger:
    def __init__(self, name: str = "velibnow"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)


def get_engine(db_url: str):
    """Crée et retourne un moteur SQLAlchemy pour la base de données."""
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        raise ConnectionError(f"Erreur de connexion à la base : {e}")


def current_timestamp() -> str:
    """Retourne l'horodatage actuel au format ISO."""
    return datetime.utcnow().isoformat()


def safe_div(a, b):
    """Division sécurisée : évite les erreurs en cas de division par zéro."""
    try:
        return a / b if b != 0 else 0
    except TypeError:
        return 0


# Exemple d'utilisation
if __name__ == "__main__":
    log = Logger()
    log.info("Démarrage de l'application VélibNow...")
    print(f"Horodatage actuel : {current_timestamp()}")
    print(f"Division sécurisée : {safe_div(10, 0)}")
