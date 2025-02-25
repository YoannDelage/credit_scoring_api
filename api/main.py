import os
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import mlflow
import logging
import uvicorn

# URI de suivi pour pointer vers serveur MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# init API FastAPI
app = FastAPI()

# Config des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Clé d'API (remplacer par la même clé utilisée dans l'application Streamlit)
API_KEY = os.getenv("API_KEY", "HRKU-b32a3477-3def-45ff-af77-23caa503e3fc")  # On récupère la clé depuis l'env

# Fonction pour charger le modèle
def load_model(model_name: str):
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de chargement du modèle : {str(e)}")

# Fonction de prédiction
def predict(model, features):
    if not isinstance(features, pd.DataFrame):
        df = pd.DataFrame(features)
    else:
        df = features
    
    prediction = model.predict(df)
    
    if hasattr(prediction, "__len__") and len(prediction) == 1:
        return prediction[0]
    return prediction

# Chargement du modèle
model = load_model("LGBM_smoted_tuned_trained/1")
if model is None:
    raise HTTPException(status_code=500, detail="Le modèle n'a pas pu être chargé.")

# Structure attendue par l'API
class InputData(BaseModel):
    SK_ID_CURR: int

# Fonction pour vérifier la clé d'API
def verify_api_key(api_key: str):
    logger.debug(f"Clé API reçue: {api_key}")
    if api_key != API_KEY:
        logger.error(f"Clé API invalide: {api_key}")
        raise HTTPException(status_code=403, detail="Clé d'API invalide")

    else:
        logger.debug("Clé API valide")  # Si la clé est valide, on log ce message


@app.get("/")
def home():
    return {"message": "API de scoring connectée à MLflow !"}

@app.post("/predict")
async def predict_api(data: InputData, request: Request, api_key: str = Header(None)):
    # Vérification de la clé d'API
    if api_key is None:
        raise HTTPException(status_code=403, detail="Clé d'API manquante")
    verify_api_key(api_key)

    try:
        logger.debug(f"Requête reçue: {data}")
        
        file_path = '../df_test.csv' 
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Fichier df_test.csv introuvable.")

        df_test = pd.read_csv(file_path)

        if 'SK_ID_CURR' not in df_test.columns:
            raise HTTPException(status_code=400, detail="La colonne 'SK_ID_CURR' est manquante dans df_test.csv.")
        
        individual = df_test[df_test['SK_ID_CURR'] == data.SK_ID_CURR]
        logger.debug(f"Individu trouvé: {individual}")

        if individual.empty:
            raise HTTPException(status_code=404, detail="Identifiant non trouvé dans le DataFrame")

        features = individual.loc[:, individual.columns != 'SK_ID_CURR']
        logger.debug(f"Features extraites: {features}")

        if features.shape[0] != 1:
            raise HTTPException(status_code=400, detail="Les features ont une forme incorrecte.")
        
        logger.debug(f"Shape des features avant prédiction: {features.shape}")
        
        prediction = predict(model, features)
        logger.debug(f"Prédiction du modèle: {prediction}")

        if prediction not in [0, 1]:
            raise HTTPException(status_code=500, detail="Prédiction invalide retournée par le modèle.")

        result = "Crédit accordé" if prediction == 0 else "Crédit refusé"

        return {"prediction": int(prediction), "resultat": result}

    except HTTPException as e:
        logger.error(f"Erreur HTTP: {e.detail}")
        return {"error": e.detail}
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        return {"error": f"Erreur dans les données entrées: {str(e)}"}
    except Exception as e:
        logger.error(f"Erreur inconnue: {str(e)}")
        return {"error": f"Une erreur est survenue: {str(e)}"}

# Si le script est exécuté directement, lancer l'application sur le bon port pour Heroku
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Prend le port défini par Heroku
    uvicorn.run(app, host="0.0.0.0", port=port)