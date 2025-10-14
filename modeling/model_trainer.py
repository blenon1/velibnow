from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib

class ModelTrainer:
    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path

    def train(self, df):
        X = df[["ratio_occ", "capacity", "is_holiday", "temperature_2m"]]
        y = (df["numbikesavailable"] == 0).astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model = GradientBoostingClassifier().fit(X_train, y_train)
        print("AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))
        joblib.dump(model, self.model_path)
        return model
