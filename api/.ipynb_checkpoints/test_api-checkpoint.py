import pandas as pd
import numpy as np
import requests

# Charger ton fichier CSV dans un DataFrame
df_test = pd.read_csv("C:/Users/ydela/Desktop/P7 Start/credit_scoring_api/df_test.csv")

# Remplacer les NaN par 0 (ou utiliser .fillna() pour une autre valeur)
df_test = df_test.fillna(0)

# Convertir toutes les colonnes en types natifs (int ou float)
df_test = df_test.applymap(lambda x: int(x) if isinstance(x, np.int64) else float(x))

# URL de ton API
url = "http://127.0.0.1:8000/predict"

# Tester une seule ligne (première ligne de df_test)
features = df_test.iloc[0].tolist()  # Prendre la première ligne pour l'exemple

# Préparer les données à envoyer
data = {"features": features}

# Faire la requête POST
response = requests.post(url, json=data)

# Afficher la réponse
print("Prédiction pour l'individu 0 : ", response.json())

# Tester plusieurs lignes (facultatif)
# Pour chaque ligne de df_test, on envoie les features et on affiche la prédiction
for index, row in df_test.iterrows():
    features = row.tolist()  # Convertir la ligne en liste
    data = {"features": features}
    
    response = requests.post(url, json=data)
    
    # Afficher la réponse pour chaque ligne
    print(f"Prédiction pour l'individu {index}: {response.json()}")
