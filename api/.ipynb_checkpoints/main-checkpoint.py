import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import logging
import uvicorn
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Config des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Init API FastAPI
app = FastAPI()

# Ajout de CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction pour charger le DataFrame
def load_dataframe():
    try:
        # Chemin absolu basé sur le répertoire courant
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'df_test_reduit.csv')
        
        # Si le fichier n'est pas à cet emplacement, essayons d'autres emplacements
        if not os.path.exists(file_path):
            file_path = os.path.join(os.path.dirname(base_dir), 'df_test_reduit.csv')
        
        if not os.path.exists(file_path):
            # Dernier recours: chercher partout dans le répertoire courant et ses sous-dossiers
            for root, dirs, files in os.walk(os.path.dirname(base_dir)):
                if 'df_test_reduit.csv' in files:
                    file_path = os.path.join(root, 'df_test_reduit.csv')
                    break
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier df_test_reduit.csv introuvable.")
            
        logger.info(f"Chargement du fichier: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du DataFrame: {str(e)}")
        raise

# Fonction pour charger le modèle
def load_model():
    try:
        # Chemin absolu basé sur le répertoire courant
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'LGBM_TTS.pkl')
        
        # Si le fichier n'est pas à cet emplacement, essayons d'autres emplacements
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(base_dir), 'LGBM_TTS.pkl')
        
        if not os.path.exists(model_path):
            # Dernier recours: chercher partout dans le répertoire courant et ses sous-dossiers
            for root, dirs, files in os.walk(os.path.dirname(base_dir)):
                if 'LGBM_TTS.pkl' in files:
                    model_path = os.path.join(root, 'LGBM_TTS.pkl')
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle LGBM_TTS.pkl introuvable.")
            
        logger.info(f"Chargement du modèle: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

# Structure attendue par l'API
class InputData(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def home():
    return {"message": "API de scoring crédit connectée !"}

@app.post("/predict")
async def predict_api(data: InputData):
    try:
        logger.debug(f"Requête reçue: {data}")
        
        # Chargement du DataFrame
        df_test_reduit = load_dataframe()

        if 'SK_ID_CURR' not in df_test_reduit.columns:
            raise HTTPException(status_code=400, detail="La colonne 'SK_ID_CURR' est manquante dans df_test_reduit.csv.")
        
        individual = df_test_reduit[df_test_reduit['SK_ID_CURR'] == data.SK_ID_CURR]
        logger.debug(f"Individu trouvé: {not individual.empty}")

        if individual.empty:
            raise HTTPException(status_code=404, detail=f"Identifiant {data.SK_ID_CURR} non trouvé dans le DataFrame")

        # Extrait les features
        features = individual.drop('SK_ID_CURR', axis=1)
        logger.debug(f"Shape des features avant prédiction: {features.shape}")
        
        # Option 1: Continuer à utiliser une prédiction aléatoire temporaire
        # si vous n'avez pas encore ajouté le modèle à votre dépôt
        import random
        prediction_temp = random.choice([0, 1])
        logger.debug(f"Prédiction temporaire: {prediction_temp}")
        
        # Option 2: Charger et utiliser le modèle joblib
        # Décommentez ce bloc et commentez le bloc "prédiction aléatoire" quand vous aurez ajouté le modèle
        """
        # Chargement du modèle
        model = load_model()
        
        # Faire la prédiction
        prediction_proba = model.predict_proba(features)[:, 1][0]
        logger.debug(f"Probabilité de défaut: {prediction_proba}")
        
        # Utiliser le seuil optimal déterminé lors de l'entraînement (à ajuster selon votre cas)
        threshold = 0.5  # à remplacer par votre seuil métier optimisé
        prediction = 1 if prediction_proba > threshold else 0
        """
        
        # Pour l'instant, utilisons la prédiction temporaire
        prediction = prediction_temp
        
        result = "Crédit refusé" if prediction == 1 else "Crédit accordé"

        # Pour une API plus complète, vous pourriez ajouter:
        # - La probabilité de défaut
        # - Les principales features qui ont influencé la décision
        # - Des informations supplémentaires sur le client

        return {
            "prediction": int(prediction), 
            "resultat": result
            # Autres informations que vous voudriez ajouter ici
        }

    except HTTPException as e:
        logger.error(f"Erreur HTTP: {e.detail}")
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur dans les données entrées: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inconnue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur est survenue: {str(e)}")

# Si le script est exécuté directement, lancer l'application sur le bon port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Port par défaut pour Render
    uvicorn.run(app, host="0.0.0.0", port=port)
