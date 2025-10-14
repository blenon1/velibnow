import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Charge le modèle entraîné depuis un fichier."""
        try:
            model = joblib.load(self.model_path)
            print(f"✅ Modèle chargé depuis {self.model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Modèle introuvable : {self.model_path}")

    def prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prépare les features nécessaires pour la prédiction."""
        required_features = ["ratio_occ", "capacity", "is_holiday", "temperature_2m"]
        for f in required_features:
            if f not in df.columns:
                raise ValueError(f"Colonne manquante : {f}")
        return df[required_features]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fait les prédictions et ajoute la probabilité au DataFrame."""
        X = self.prepare_input(df)
        df["proba"] = self.model.predict_proba(X)[:, 1]
        df["pred_rupture"] = (df["proba"] > 0.5).astype(int)
        print(f"✅ Prédictions effectuées sur {len(df)} lignes.")
        return df

    def save_predictions(self, df: pd.DataFrame, path: str = "predictions.csv"):
        df.to_csv(path, index=False)
        print(f"💾 Prédictions sauvegardées dans {path}")


if __name__ == "__main__":
    # Exemple d'utilisation
    predictor = Predictor(model_path="model.pkl")
    sample_data = pd.DataFrame({
        "ratio_occ": [0.3, 0.9, 0.1],
        "capacity": [20, 25, 15],
        "is_holiday": [0, 1, 0],
        "temperature_2m": [12.5, 8.3, 14.1]
    })
    predictions = predictor.predict(sample_data)
    print(predictions)
    predictor.save_predictions(predictions)
