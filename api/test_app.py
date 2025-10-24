# test_app.py
import pytest
from unittest.mock import patch, MagicMock
import requests
import json
import sys
import os

# Ajouter le chemin du répertoire parent pour importer app.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importer la fonction à tester depuis app.py
from app import get_prediction

class TestAppFunctions:
    
    def test_get_prediction_empty_id(self):
        """Tester que la fonction gère correctement un ID vide"""
        result = get_prediction('')
        assert result == "Erreur : L'ID du client ne peut pas être vide."
    
    def test_get_prediction_non_integer_id(self):
        """Tester que la fonction gère correctement un ID non entier"""
        result = get_prediction('abc')
        assert result == "Erreur : L'ID du client doit être un entier valide."
    
    @patch('requests.post')
    def test_get_prediction_successful_request(self, mock_post):
        """Tester une requête API réussie avec un ID valide"""
        # Configurer le mock pour simuler une réponse réussie
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'prediction': 0,
            'resultat': 'Crédit accordé'
        }
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec un ID valide
        result = get_prediction('123456')
        
        # Vérifier que la fonction a appelé l'API avec les bons paramètres
        mock_post.assert_called_once_with(
            'https://credit-scoring-api-8lkh.onrender.com/predict',
            json={'SK_ID_CURR': 123456},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        # Vérifier le résultat
        assert result == {'prediction': 0, 'resultat': 'Crédit accordé'}
    
    @patch('requests.post')
    def test_get_prediction_api_error(self, mock_post):
        """Tester la gestion d'une erreur de l'API"""
        # Configurer le mock pour simuler une erreur de l'API
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec un ID valide
        result = get_prediction('123456')
        
        # Vérifier que le message d'erreur est correct
        assert "Erreur dans la prédiction : Code 404" in result
    
    @patch('requests.post')
    def test_get_prediction_connection_error(self, mock_post):
        """Tester la gestion d'une erreur de connexion à l'API"""
        # Configurer le mock pour simuler une erreur de connexion
        mock_post.side_effect = requests.exceptions.ConnectionError("Impossible de se connecter au serveur")
        
        # Appeler la fonction avec un ID valide
        result = get_prediction('123456')
        
        # Vérifier que le message d'erreur est correct
        assert "Erreur de connexion à l'API : " in result
    
    @patch('requests.post')
    def test_get_prediction_malformed_response(self, mock_post):
        """Tester la gestion d'une réponse mal formée de l'API"""
        # Configurer le mock pour simuler une réponse mal formée
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"some_other_key": "value"}
        mock_response.text = json.dumps(mock_response.json.return_value)
        mock_post.return_value = mock_response
        
        # Appeler la fonction avec un ID valide
        result = get_prediction('123456')
        
        # Vérifier que le message d'erreur est correct
        assert result == "Erreur : La réponse de l'API est mal formée."