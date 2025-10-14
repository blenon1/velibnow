import requests
import json
import pandas as pd
from datetime import datetime

class APITester:
    def __init__(self, apis: dict, timeout: int = 5):
        """
        apis : dictionnaire {nom_api: url}
        timeout : délai en secondes avant expiration de la requête
        """
        self.apis = apis
        self.timeout = timeout
        self.results = []

    def test_all(self, save_csv: bool = True):
        print("🔍 Test automatique des endpoints API\n")
        for name, url in self.apis.items():
            self.test_one(name, url)

        if save_csv:
            self._save_results()

    def test_one(self, name, url):
        print(f"➡️ {name}")
        print(f"   URL : {url}")
        result = {
            "api_name": name,
            "url": url,
            "status_code": None,
            "content_type": None,
            "keys": None,
            "response_time_s": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            r = requests.get(url, timeout=self.timeout)
            result["status_code"] = r.status_code
            result["response_time_s"] = round(r.elapsed.total_seconds(), 2)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "unknown")
            result["content_type"] = content_type
            print(f"   ✅ Statut HTTP : {r.status_code} ({result['response_time_s']}s)")
            print(f"   📦 Type : {content_type}")

            if "json" in content_type:
                try:
                    data = r.json()
                    print(f"   🔍 Aperçu du JSON :")
                    keys = self._inspect_json(data)
                    result["keys"] = keys
                except json.JSONDecodeError:
                    print("   ❌ Erreur : réponse non décodable en JSON")
                    result["error"] = "Invalid JSON"
            else:
                print(f"   ⚠️ Contenu non JSON (texte tronqué) : {r.text[:200]}...")

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erreur : {e}")
            result["error"] = str(e)

        self.results.append(result)
        print("-" * 80)

    def _inspect_json(self, data):
        """
        Affiche les 3 premières clés ou structures principales et les retourne.
        """
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            print(f"     → Clés principales : {keys}")
            for k in keys:
                v = data[k]
                if isinstance(v, (list, dict)):
                    print(f"       • {k}: type {type(v).__name__}, taille {len(v)}")
            return keys
        elif isinstance(data, list):
            print(f"     → Liste de {len(data)} éléments")
            if len(data) > 0 and isinstance(data[0], dict):
                keys = list(data[0].keys())[:5]
                print(f"       • Clés de l’objet 0 : {keys}")
                return keys
            return ["list"]
        else:
            print(f"     → Type inattendu : {type(data).__name__}")
            return [type(data).__name__]

    def _save_results(self):
        """Sauvegarde les résultats dans un fichier CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv("api_test_results.csv", index=False)
        print("💾 Résultats enregistrés dans api_test_results.csv\n")

# -----------------------------------------------------

if __name__ == "__main__":
    apis = {
        "Jours fériés": "https://calendrier.api.gouv.fr/jours-feries/metropole.json",
        "Vacances scolaires": "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records?limit=5",
        "Vélib disponibilité": "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
    }

    tester = APITester(apis)
    tester.test_all()
