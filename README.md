## Structure du projet 
```bash
velibnow-historicalManager/
│  .gitignore                 # Fichiers/dossiers ignorés par Git
│  conception.ipynb           # Notebook d’exploration/POC
│  config.py                  # Variables globales (ex: URLs, chemins, clés)
│  docker-compose.yaml        # Orchestration Docker (backend + dashboard + volumes)
│  main.py                    # Script d’ingestion orchestré (charge & sauvegarde en DB)
│  README.md                  # Doc du projet
│  requirements.txt           # Dépendances Python racine
│
├─ backend/
│  │  Dockerfile              # Image backend (FastAPI + modèles)
│  │  predictor.py            # Classe Predictor pour prédire (inférence : features + predict_proba)
│  │  requirements.txt        # Dépendances backend
│  │  server.py               # API FastAPI (/health, /stations, /predict…)
│  │
│  └─ data/
│        stations.csv         # Liste stations (fallback/échantillon pour tests)
│
├─ core/
│  │  database.py             # DatabaseManager : connexion SQLAlchemy + to_sql/read_sql
│  │  docker-compose.yaml     # Compose spécifique DB (à lancer pour charger les données avant entrainement)
│
├─ dashboard/
│  │  app.py                  # Front Streamlit (UI) : paramètres + carte + tableaux
│  │  data_api.py             # Client HTTP du front (appelle le backend)
│  │  Dockerfile              # Image dashboard (Streamlit)
│  │  requirements.txt        # Dépendances dashboard
│
├─ data/
│  │  holidays_api.py         # Récupère jours fériés & vacances (API)
│  │  velib_api.py            # Accès API temps réel stations Vélib (info)
│  │  velib_historical_csv.py # Reader CSV historique Vélib (statuts)
│  │  weather_api.py          # Accès API météo (temps réel/complément)
│  │  weather_historical_csv.py # Reader CSV historique météo (standardisé)
│  │
│  └─ dataset/ # Source de données à insérer (historical_meteo.csv & historique_stations.csv)
│
├─ features/
│  │  feature_builder.py      # Construction du jeu de features (time flags, merges…)
│
└─ modeling/
   │  model_trainer.py        # Entraînement : split, pipeline sklearn, save joblib
   │
   └─ models_storage/
         model_empty.joblib   # Modèle pour prédire “station vide”
         model_full.joblib    # Modèle pour prédire “station pleine”
```

## Setup de l'environnement
```bash
python3.12 -m venv .venv
source .venv/bin/activate # Linux
.\.venv\Scripts\activate # Window
pip install -r requirements.txt
```

## Setup de la base de données
**Prérequis** : Avoir docker installé sur la machine d'éxécution  
```bash
cd core
docker compose up -d
cd .. # Revenir à la racine du projet
```

## Traitement des données et entrainement du model
**Prérequis** : S'assurer que les données sources (.csv) soient disponibles dans /data/dataset  
```bash
python main.py # Exécute le code étape par étape
```
Une fois les données traitées et entrainées, les models seront stockés dans /modeling/models_storage

## Utilisation des models pour prédiction
**Prérequis** : Avoir docker installé sur la machine d'éxécution 
A la racine du projet :  
```bash 
docker compose up -d
``` 
Cela va démarer un backend (fastapi) du dossier ./backend ainsi que un (frontend) streamlit dans le dossier ./dashboard