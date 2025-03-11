import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import logging
import uvicorn
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

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
        
        # Chargement du modèle
        model = load_model()
        
        # Faire la prédiction
        prediction_proba = model.predict_proba(features)[:, 1][0]
        logger.debug(f"Probabilité de défaut: {prediction_proba}")
        
        # Utiliser le seuil optimal déterminé lors de l'entraînement
        threshold = 0.5  # à remplacer par votre seuil métier optimisé
        prediction = 1 if prediction_proba > threshold else 0
        
                    # Extraction de l'importance des features
        try:
            # Pour LightGBM, l'importance des features est disponible dans feature_importances_
            feature_importance = model.feature_importances_
            
            # Obtention des noms des features
            feature_names = features.columns.tolist()
            
            # Création d'un dictionnaire {feature_name: importance}
            feature_importance_dict = dict(zip(feature_names, feature_importance))
            
            # Tri des features par importance décroissante
            sorted_feature_importance = sorted(
                feature_importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Extraction des 15 features les plus importantes pour éviter de surcharger la réponse
            top_features = sorted_feature_importance[:15]
            
            feature_importance_data = {
                "feature_names": [feature[0] for feature in top_features],
                "importance_values": [float(feature[1]) for feature in top_features]  # Conversion en float pour sérialisation JSON
            }
            logger.debug(f"Feature importance extraite: {feature_importance_data}")
            
            # Calculer la contribution de chaque feature pour le graphique en cascade
            try:
                # Obtenir les contributions individuelles (SHAP values ou attribution similaire)
                # Note: nous simulons ces valeurs car l'extraction réelle nécessiterait 
                # un modèle spécifiquement configuré ou une librairie SHAP
                
                # Valeur de base (moyenne des prédictions)
                base_value = 0.5  # Valeur neutre pour le prêt
                
                # Obtenir les valeurs normalisées des features du client
                client_features = features.iloc[0].values
                
                # Simuler la contribution de chaque feature (en pratique, utiliser SHAP)
                # Pour une simulation réaliste, on utilise l'importance de la feature
                # et on ajoute un facteur basé sur la valeur du client
                feature_contributions = []
                for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                    # Normaliser la valeur du client entre -1 et 1 (simple simulation)
                    normalized_value = (client_features[i] - features[name].mean()) / (features[name].std() + 1e-8)
                    normalized_value = max(min(normalized_value, 3), -3) / 3  # Limiter les valeurs extrêmes
                    
                    # La contribution est basée sur l'importance et la valeur normalisée
                    # Le signe indique si la feature contribue positivement ou négativement
                    contribution = importance * normalized_value * 0.1
                    
                    feature_contributions.append((name, contribution))
                
                # Trier par valeur absolue de contribution
                sorted_contributions = sorted(
                    feature_contributions, 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                # Prendre les top N contributions (positives et négatives)
                top_contributions = sorted_contributions[:10]
                
                waterfall_data = {
                    "feature_names": [feature[0] for feature in top_contributions],
                    "contribution_values": [float(feature[1]) for feature in top_contributions],
                    "base_value": float(base_value)
                }
                
                feature_importance_data["waterfall"] = waterfall_data
                logger.debug(f"Waterfall data générée: {waterfall_data}")
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des contributions pour le waterfall: {str(e)}")
                # Ne pas échouer complètement si cette partie échoue
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de l'importance des features: {str(e)}")
            feature_importance_data = {"error": str(e)}
        
        result = "Crédit refusé" if prediction == 1 else "Crédit accordé"

        return {
            "prediction": int(prediction), 
            "resultat": result,
            "proba": float(prediction_proba),
            "feature_importance": feature_importance_data
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
