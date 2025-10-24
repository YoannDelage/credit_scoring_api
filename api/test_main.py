# test_main.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Ajouter le chemin du répertoire parent pour importer main.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importer l'application FastAPI et les fonctions à tester
from main import app, load_dataframe, load_model

# Créer un client de test
client = TestClient(app)

class TestMainFunctions:
    
    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('os.walk')
    def test_load_dataframe_success(self, mock_walk, mock_read_csv, mock_exists):
        """Tester le chargement réussi du DataFrame"""
        # Simuler que le fichier existe
        mock_exists.return_value = True
        
        # Créer un DataFrame de test
        test_df = pd.DataFrame({
            'SK_ID_CURR': [100001, 100002, 100003],
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6]
        })
        mock_read_csv.return_value = test_df
        
        # Appeler la fonction
        result = load_dataframe()
        
        # Vérifier que read_csv a été appelé
        mock_read_csv.assert_called_once()
        
        # Vérifier que le DataFrame retourné est celui attendu
        pd.testing.assert_frame_equal(result, test_df)
    
    @patch('os.path.exists')
    @patch('os.walk')
    def test_load_dataframe_file_not_found(self, mock_walk, mock_exists):
        """Tester la gestion d'erreur quand le fichier est introuvable"""
        # Simuler que le fichier n'existe pas
        mock_exists.return_value = False
        mock_walk.return_value = iter([])
        
        # Vérifier que l'exception est levée
        with pytest.raises(FileNotFoundError):
            load_dataframe()
    
    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('os.walk')
    def test_load_model_success(self, mock_walk, mock_joblib_load, mock_exists):
        """Tester le chargement réussi du modèle"""
        # Simuler que le fichier existe
        mock_exists.return_value = True
        
        # Créer un modèle de test
        test_model = MagicMock()
        mock_joblib_load.return_value = test_model
        
        # Appeler la fonction
        result = load_model()
        
        # Vérifier que joblib.load a été appelé
        mock_joblib_load.assert_called_once()
        
        # Vérifier que le modèle retourné est celui attendu
        assert result == test_model
    
    @patch('os.path.exists')
    @patch('os.walk')
    def test_load_model_file_not_found(self, mock_walk, mock_exists):
        """Tester la gestion d'erreur quand le fichier du modèle est introuvable"""
        # Simuler que le fichier n'existe pas
        mock_exists.return_value = False
        mock_walk.return_value = iter([])
        
        # Vérifier que l'exception est levée
        with pytest.raises(FileNotFoundError):
            load_model()

class TestAPIEndpoints:
    
    def test_home_endpoint(self):
        """Tester l'endpoint racine de l'API"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "API de scoring crédit connectée !"}
    
    @patch('main.load_dataframe')
    @patch('main.load_model')
    def test_predict_api_success(self, mock_load_model, mock_load_dataframe):
        """Tester l'endpoint de prédiction avec succès"""
        # Créer un DataFrame de test avec l'ID demandé
        test_df = pd.DataFrame({
            'SK_ID_CURR': [100001],
            'Feature1': [1],
            'Feature2': [2]
        })
        
        # Créer un modèle mock qui retourne toujours la même prédiction
        test_model = MagicMock()
        test_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        test_model.feature_importances_ = np.array([0.5, 0.5])
        
        # Configurer les mocks
        mock_load_dataframe.return_value = test_df
        mock_load_model.return_value = test_model
        
        # Faire la requête de test
        response = client.post("/predict", json={"SK_ID_CURR": 100001})
        
        # Vérifier la réponse
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["prediction"] == 1
        assert response_json["resultat"] == "Crédit refusé"
        assert "proba" in response_json
        assert "feature_importance" in response_json
    
    @patch('main.load_dataframe')
    def test_predict_api_id_not_found(self, mock_load_dataframe):
        """Tester l'endpoint de prédiction quand l'ID n'est pas trouvé"""
        # Créer un DataFrame de test sans l'ID demandé
        test_df = pd.DataFrame({
            'SK_ID_CURR': [100001, 100002],
            'Feature1': [1, 2],
            'Feature2': [3, 4]
        })
        
        # Configurer le mock
        mock_load_dataframe.return_value = test_df
        
        # Faire la requête de test avec un ID non présent
        response = client.post("/predict", json={"SK_ID_CURR": 999999})
        
        # Vérifier la réponse
        assert response.status_code == 404
        assert "non trouvé dans le DataFrame" in response.json()["detail"]
    
    @patch('main.load_dataframe')
    def test_predict_api_missing_column(self, mock_load_dataframe):
        """Tester l'endpoint de prédiction quand la colonne SK_ID_CURR est manquante"""
        # Créer un DataFrame de test sans la colonne SK_ID_CURR
        test_df = pd.DataFrame({
            'Feature1': [1, 2],
            'Feature2': [3, 4]
        })
        
        # Configurer le mock
        mock_load_dataframe.return_value = test_df
        
        # Faire la requête de test
        response = client.post("/predict", json={"SK_ID_CURR": 100001})
        
        # Vérifier la réponse
        assert response.status_code == 400
        assert "La colonne 'SK_ID_CURR' est manquante" in response.json()["detail"]