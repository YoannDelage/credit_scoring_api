from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import requests
from model import load_model, predict  # import du fichier model.py

# init FASTapi
app = FastAPI()

# chgt modele mlflow
model = load_model("mon_modele_en_production")  # Utilise le nom de ton modèle ici

# des format attendu
class InputData(BaseModel):
    features: List[float]  # Liste des valeurs numériques représentant un individu

@app.get("/")
def home():
    return {"message": "API de scoring connectée à MLflow !"}

@app.post("/predict")
def predict_api(data: InputData):
    try:
        # Utiliser la fonction de prédiction définie dans model.py
        prediction = predict(model, data.features)

        # Mapping du résultat
        if prediction == 0.0:
            resultat = "Crédit accordé"
        elif prediction == 1.0:
            resultat = "Crédit refusé"
        else:
            resultat = "Prédiction inconnue"

        return {"prediction": int(prediction[0]), "resultat": resultat}

    except Exception as e:
        return {"error": str(e)}

