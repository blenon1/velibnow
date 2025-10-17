from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import joblib


class VelibSimpleModel:
    """
    Modèle simple et lisible pour prédire si une station sera vide/pleine.

    Principes :
    - Utilise uniquement les colonnes déjà présentes dans votre DataFrame d'exemple
      (pas de météo, pas de mapping).
    - Encodage temporel cyclique (à partir de 'time').
    - Imputation légère + standardisation pour les numériques, OHE pour station_id.
    - API courte: fit / predict_proba / predict_label / save / load
    - Paramètres regroupés en dict (to_config / from_config).
    """

    # Colonnes de base attendues (selon votre DataFrame d'entrée)
    BASE_NUM_SCALED: List[str] = [
        # features temporelles
        "hour_ssin", "hour_ccos", "dow_sin", "dow_cos", "month",
    ]
    BASE_BIN_PASSTHROUGH: List[str] = [
        "holiday_flag", "is_weekend", "operative",
        "pluie", "vent", "soleil", "nuage",
    ]
    BASE_CATEG: List[str] = ["station_name"]

    def __init__(
        self,
        target_col: str = "target_empty",        # ou "target_full"
        model_type: str = "gb",                  # "gb" ou "logit"
        timezone: str = "Europe/Paris",
        random_state: int = 42,
        test_size: float = 0.2,
        verbose: bool = True,
    ):
        print("[__init__] → Début initialisation du modèle")
        self.target_col = target_col
        self.model_type = model_type
        self.timezone = timezone
        self.random_state = random_state
        self.test_size = test_size
        self.verbose = verbose

        # Objets entraînés
        self.feature_list_: List[str] = []
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.pipeline_: Optional[Pipeline] = None
        print("[__init__] ✓ Fin initialisation du modèle")

    # ------------- Helpers ---------------

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
        print("[_ensure_columns] → Vérification des colonnes requises...")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        print("[_ensure_columns] ✓ Toutes les colonnes sont présentes.")

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute hour/dow/month + encodage cyclique à partir de la colonne 'time'.
        On ne dépend pas des colonnes 'hour'/'day_of_week' existantes pour garder la logique simple et robuste.
        """
        print("[_add_time_features] → Début génération des features temporelles...")
        if "time" not in df.columns:
            raise ValueError("La colonne 'time' est requise.")
        ts = pd.to_datetime(df["time"], utc=True).dt.tz_convert(self.timezone)

        out = df.copy()
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.weekday     # 0 = lundi
        out["month"] = ts.dt.month

        out["hour_ssin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_ccos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7)
        out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7)
        print("[_add_time_features] ✓ Features temporelles ajoutées.")
        return out

    def _build_preprocessor(self) -> None:
        """
        Préprocesseur:
        - Numériques (imputation médiane + standardisation)
        - Binaires (imputation la plus fréquente, pas de scaling)
        - Catégorielles (OHE handle_unknown='ignore')
        """
        print("[_build_preprocessor] → Construction du préprocesseur...")
        # Gestion valeurs manquantes + standardisation
        num_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ])
        # Gestion valeurs manquantes (pas scaler pour garder 0/1 lisible)
        bin_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
        ])
        # Transforme variables catégorielles en binaire
        cat_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", OneHotEncoder(
                handle_unknown="ignore",
                dtype=np.float32,
                sparse_output=True,
            )),
        ])

        # Liste des features finales
        self.feature_list_ = (
            self.BASE_NUM_SCALED
            + self.BASE_BIN_PASSTHROUGH
            + self.BASE_CATEG
        )

        # Applique les 3 pipelines de transformation + drop autres colonnes
        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.BASE_NUM_SCALED),
                ("bin", bin_pipe, self.BASE_BIN_PASSTHROUGH),
                ("cat", cat_pipe, self.BASE_CATEG),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        print("[_build_preprocessor] ✓ Préprocesseur construit.")

    def _prepare_features(
        self, df: pd.DataFrame, with_target: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        - Ajoute les features temporelles
        - Vérifie la présence des colonnes attendues
        - Renvoie X (et y si with_target)
        """
        print("[_prepare_features] → Préparation des features...")
        # Ajoute les features temporelles
        df2 = self._add_time_features(df)

        # Construire le préprocesseur si pas encore fait
        if self.preprocessor_ is None:
            self._build_preprocessor()

        # Vérifier les colonnes d'entrée attendues
        needed = (
            set(self.BASE_NUM_SCALED + self.BASE_BIN_PASSTHROUGH + self.BASE_CATEG)
            - {"hour_ssin", "hour_ccos", "dow_sin", "dow_cos", "month"}  # créées ici
        )
        self._ensure_columns(df2, sorted(needed))

        X = df2[self.feature_list_] # Liste de colonne créée dans _build_preprocessor()

        y = None
        if with_target: # Vérification de la présence de la colonne cible (à prédire)
            if self.target_col not in df2.columns:
                raise ValueError(f"Colonne cible manquante: '{self.target_col}'")
            y = df2[self.target_col].astype(int)

        # Renvoi les colonnes de paramètre ainsi que la colonne cible 
        print("[_prepare_features] ✓ Features préparées.")
        return X, y

    # ------------- API publique ---------------

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne le modèle choisi et renvoie des métriques simples (AUC).
        """
        print("[fit] → Début entraînement du modèle...")
        
        self._build_preprocessor()# (ré)initialise un préprocesseur propre
        X, y = self._prepare_features(df, with_target=True) # Préparation des features

        # Choix du classifieur
        if self.model_type == "logit":
            classifier = LogisticRegression(max_iter=300, solver="saga", verbose=1, random_state=self.random_state)
        else:
            classifier = GradientBoostingClassifier(random_state=self.random_state, verbose=1)

        # Assemblage préprocesseur + classifieur
        self.pipeline_ = Pipeline(steps=[("preprocessor", self.preprocessor_), ("classifier", classifier)])

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print("[fit] → Entraînement en cours...")
        self.pipeline_.fit(X_tr, y_tr)

        print("[fit] ✓ Entraînement terminé. Évaluation en cours...")

        proba = self.pipeline_.predict_proba(X_va)[:, 1]
        y_pred = (proba >= 0.5).astype(int)  # seuil simple

        acc = accuracy_score(y_va, y_pred)
        f1  = f1_score(y_va, y_pred)

        metrics = {
            "val_accuracy": float(acc),
            "val_f1": float(f1),
            "n_samples_train": int(len(X_tr)),
            "n_samples_val": int(len(X_va)),
        }

        if self.verbose:
            print(f"[{self.target_col}] {self.model_type.upper()}  "
                f"ACC={metrics['val_accuracy']:.3f}  F1={metrics['val_f1']:.3f}  "
                f"(train={metrics['n_samples_train']}, val={metrics['n_samples_val']})")


        print("[fit] ✓ Fin de l'entraînement et des métriques.")
        return metrics

    def save(self, path: str) -> None:
        """Sauvegarde le pipeline complet (prétraitement + modèle)."""
        print("[save] → Sauvegarde du modèle...")
        check_is_fitted(self.pipeline_, "named_steps")
        joblib.dump(self.pipeline_, path)
        print("[save] ✓ Modèle sauvegardé.")

    # --------- Config dict ---------

    @classmethod
    def from_config(cls, cfg: Dict) -> "VelibSimpleModel":
        print("[from_config] → Création du modèle depuis un dictionnaire de config...")
        model = cls(**cfg)
        print("[from_config] ✓ Modèle créé depuis la config.")
        return model

    def to_config(self) -> Dict:
        print("[to_config] → Export de la configuration du modèle...")
        cfg = {
            "target_col": self.target_col,
            "model_type": self.model_type,
            "timezone": self.timezone,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "verbose": self.verbose,
        }
        print("[to_config] ✓ Configuration exportée.")
        return cfg
