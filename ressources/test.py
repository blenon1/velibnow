# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Définir la période
start = datetime(2020, 1, 1)
end = datetime(2024, 12, 31)

# Créer un Point pour Paris, France
paris = Point(48.8566, 2.3522, 35)  # latitude, longitude, altitude (m)

# Récupérer les données journalières
data = Daily(paris, start, end)
data = data.fetch()

# Afficher les 5 premières lignes pour vérification
print(data.head())

# Tracer les températures moyenne, minimale et maximale
data.plot(y=['tavg', 'tmin', 'tmax'], title="Températures à Paris (2020–2024)")
plt.xlabel("Date")
plt.ylabel("Température (°C)")
plt.grid(True)
plt.show()
