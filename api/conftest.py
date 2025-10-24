# conftest.py
"""
Ce fichier contient des fixtures Pytest que nous pouvons réutiliser dans plusieurs tests.
Les fixtures sont des fonctions qui fournissent des données ou des états prédéfinis pour les tests.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_df():
    """Fixture fournissant un DataFrame d'exemple pour les tests"""
    return pd.DataFrame({
        'SK_ID_CURR': [100001, 100002, 100003],
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'Feature3': [7, 8, 9]
    })

@pytest.fixture
def mock_model():
    """Fixture fournissant un modèle mockée pour les tests"""
    model = MagicMock()
    # Configurer le modèle pour qu'il retourne des prédictions cohérentes
    model.predict_proba.return_value = np.array([
        [0.7, 0.3],  # Pour le client 1 (70% proba classe 0, 30% proba classe 1)
        [0.4, 0.6],  # Pour le client 2 (40% proba classe 0, 60% proba classe 1)
        [0.8, 0.2]   # Pour le client 3 (80% proba classe 0, 20% proba classe 1)
    ])
    return model

@pytest.fixture
def api_response_success():
    """Fixture fournissant une réponse d'API réussie"""
    return {
        'prediction': 0,
        'resultat': 'Crédit accordé'
    }

@pytest.fixture
def api_response_failure():
    """Fixture fournissant une réponse d'API pour un crédit refusé"""
    return {
        'prediction': 1,
        'resultat': 'Crédit refusé'
    }