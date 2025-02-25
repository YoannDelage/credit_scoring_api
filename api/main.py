import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging
import uvicorn

# Init API FastAPI
app = FastAPI()

# Config des logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Structure attendue par l'API
class InputData(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def home():
    return {"message": "API de scoring crédit connectée !"}

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

# Modification pour utiliser des prédictions simplifiées en attendant de faire fonctionner MLflow
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

        # Pour l'instant, utilisons une prédiction simple aléatoire
        # Remplacer ceci par votre vraie logique de prédiction quand MLflow fonctionnera
        import random
        prediction = random.choice([0, 1])
        
        result = "Crédit accordé" if prediction == 0 else "Crédit refusé"

        return {"prediction": int(prediction), "resultat": result}

    except HTTPException as e:
        logger.error(f"Erreur HTTP: {e.detail}")
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur dans les données entrées: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inconnue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur est survenue: {str(e)}")

# Si le script est exécuté directement, lancer l'application sur le bon port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Port par défaut pour Render
    uvicorn.run(app, host="0.0.0.0", port=port)
