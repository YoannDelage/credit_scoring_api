from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
from api.model import load_model, predict 
import mlflow
import os
import numpy as np
import logging

# def URI de suivi pour pointer vers serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")  

# init API FastAPI
app = FastAPI()

# config des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# chgt du modèle 
model = load_model("LGBM_smoted_tuned_trained/1")  # modele MLflow
if model is None:
    raise HTTPException(status_code=500, detail="Le modèle n'a pas pu être chargé.")

# structure attendue par l'API
class InputData(BaseModel):
    SK_ID_CURR: int  # SK_ID_CURR en entrée

@app.get("/")
def home():
    return {"message": "API de scoring connectée à MLflow !"}

@app.post("/predict")
async def predict_api(data: InputData, request: Request):
    try:
        # affiche les données reçues pour le débogage
        logger.debug(f"Requête reçue: {data}")
        
        # chgt DF test 
        file_path = '../df_test.csv' 
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Fichier df_test.csv introuvable.")

        df_test = pd.read_csv(file_path)

        # verif présence de 'SK_ID_CURR'
        if 'SK_ID_CURR' not in df_test.columns:
            raise HTTPException(status_code=400, detail="La colonne 'SK_ID_CURR' est manquante dans le fichier df_test.csv.")
        
        # recherche l'individu par son SK_ID_CURR dans le DF de test
        individual = df_test[df_test['SK_ID_CURR'] == data.SK_ID_CURR]
        logger.debug(f"Individu trouvé: {individual}")

        # verif si l'individu existe
        if individual.empty:
            raise HTTPException(status_code=404, detail="Identifiant non trouvé dans le DataFrame")

        # extract des features nécessaires (en excluant SK_ID_CURR)
        features = individual.loc[:, individual.columns != 'SK_ID_CURR']
        logger.debug(f"Features extraites: {features}")

        # verif que la shape est OK
        if features.shape[0] != 1:
            raise HTTPException(status_code=400, detail="Les features ont une forme incorrecte.")
        
        logger.debug(f"Shape des features avant prédiction: {features.shape}")
        
        # prediction avec le modele par la fonction 'predict' dans model.py
        prediction = predict(model, features)
        logger.debug(f"Prédiction du modèle: {prediction}")

        # verif de la forme de la pred
        if prediction not in [0, 1]:
            raise HTTPException(status_code=500, detail="Prédiction invalide retournée par le modèle.")

        # mapping du resultat
        result = "Crédit accordé" if prediction == 0 else "Crédit refusé"

        # retourner la réponse
        return {"prediction": int(prediction), "resultat": result}

    except HTTPException as e:
        logger.error(f"Erreur HTTP: {e.detail}")
        return {"error": e.detail}
    except Exception as e:
        logger.error(f"Erreur inconnue: {str(e)}")
        return {"error": f"Une erreur est survenue: {str(e)}"}