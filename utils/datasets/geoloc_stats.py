import pandas as pd
import numpy as np
import reverse_geocoder as rg
from sklearn.cluster import DBSCAN

# 1. Charger les données
df = pd.read_csv('/data/infraspecies/vegetables/BananaTrees/banana_trees_V5.csv')

# 2. Nettoyage des coordonnées
df['latitude'] = df['latitude'].astype(str).str.replace(',', '.')
df['longitude'] = df['longitude'].astype(str).str.replace(',', '.')

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

df = df.dropna(subset=['latitude', 'longitude'])

# Filtrage de sécurité pour les limites géographiques standards
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]

# --- PARTIE 1 : RECHERCHE DES PAYS ---
coords = list(zip(df['latitude'], df['longitude']))
print(f"Recherche des pays en cours pour {len(coords)} observations valides...")
results = rg.search(coords)

countries = set([res['cc'] for res in results])
print(f"Nombre de pays différents couverts : {len(countries)}")
print(f"Liste des codes pays : {countries}\n")

# --- PARTIE 2 : CALCUL DES LOCALITÉS (1 km) ---
# Conversion en radians requise pour la métrique de Haversine
coords_rad = np.radians(df[['latitude', 'longitude']].to_numpy())

earth_radius_km = 6371.0
epsilon = 1.0 / earth_radius_km  # Tolérance de 1 km

print(f"Lancement du clustering spatial (DBSCAN à 1 km)...")
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords_rad)

num_localities = len(set(db.labels_))
print(f"Nombre de localités distinctes : {num_localities}")