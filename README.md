# velibnow-historicalManager

Projet de bout en bout qui **ingère les historiques Vélib / météo / calendrier**, **entraîne deux modèles** (stations *vides* et *pleines*) et **expose les prédictions** via un **backend FastAPI**, consultables dans un **dashboard Streamlit** (tableaux + carte).

**But** : aider les usagers à **trouver rapidement** des stations avec **vélos disponibles au départ** et **places libres à l’arrivée**, en sélectionnant un **jour** et une **heure**. Les prédictions automatisent cette recherche pour **simplifier l’expérience utilisateur**.

---

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
## Prérequis
- **Docker** (exécuter backend + dashboard facilement)
- (Optionnel) **Python 3.12** si tu veux lancer ingestion/entraînement en local
- (Optionnel) **PostgreSQL** si tu n’utilises pas Docker pour la DB

## Setup de l’environnement (local)
```bash
python3.12 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

## Lancer la base de données (si utilisée)
```bash
cd core
docker compose up -d        # lance PostgreSQL
cd ..                       # reviens à la racine
```

## Ingestion des données → PostgreSQL
> Assure-toi d’avoir les CSV sources dans `data/dataset/`.

```bash
python main.py
```
- Charge l’historique Vélib + météo + calendrier et sauvegarde en base.
- Le DSN Postgres est lu depuis `config.py` (ex: `POSTGRES_URL`).

## Entraînement des modèles
Le script typique (voir `modeling/model_trainer.py`) :
- Charge les features (via `features/feature_builder.py`).
- Entraîne deux modèles (ex: `target_empty` et `target_full`).
- Sauvegarde dans `modeling/models_storage/`.

**Sorties attendues :**
```
modeling/models_storage/
  ├─ model_empty.joblib
  └─ model_full.joblib
```

## Lancer l’application (backend + dashboard)
À la **racine** du projet :
```bash
docker compose up -d
```
- **Backend (FastAPI)** : http://localhost:8000  
  - Endpoints utiles :
    - `GET /health` : statut
    - `GET /stations` : infos stations (via API Vélib ou fallback CSV)
    - `GET /stations/csv` : noms de stations du CSV côté serveur
    - `POST /predict` : lance les deux prédictions et renvoie un DataFrame combiné

- **Dashboard (Streamlit)** : http://localhost:8501  
  - Affiche un aperçu des stations (info), la liste des stations (CSV côté backend), lance les prédictions, affiche le tableau résultats et une **carte** pydeck (couleur ↔ proba).

## Variables & chemins importants
- **Modèles (backend)**  
  - Variables d’env : `MODEL_EMPTY_PATH`, `MODEL_FULL_PATH`  
  - Par défaut montés via le volume Compose :
    ```yaml
    ./modeling/models_storage:/models:ro
    ```
  - Dans le conteneur backend : `/models/model_empty.joblib`, `/models/model_full.joblib`

- **CSV des stations côté backend (optionnel)**  
  - Généralement `backend/data/stations.csv` (ou volume similaire)
  - L’API `/stations/csv` renvoie la liste des `station_name` utilisés pour la prédiction.

## Flux logique (résumé)
1. **Ingestion** (`main.py`) : lit CSV/API → **Postgres** (via `core/database.py`).
2. **Features** (`features/feature_builder.py`) → dataset d’entraînement.
3. **Entraînement** (`modeling/model_trainer.py`) → export `.joblib`.
4. **Backend** (`backend/server.py`) : charge les modèles, sert `/predict`.
5. **Dashboard** (`dashboard/app.py`) : appelle le backend, affiche **tableau** et **carte**.

## Dépannage rapide
- `404 /stations` : backend pas démarré, ou endpoint non exposé → relancer `docker compose up -d`.
- `python-multipart` manquant : ajoute-le dans `backend/requirements.txt`.
- Modèles introuvables : vérifie le **volume** des modèles dans `docker-compose.yaml` et les env `MODEL_EMPTY_PATH` / `MODEL_FULL_PATH`.
