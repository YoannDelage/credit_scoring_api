# Projet Credit Scoring API

## Description
Projet de scoring crédit utilisant le machine learning pour prédire l'octroi ou le refus d'un crédit. Ce projet comprend une API FastAPI et une interface Streamlit pour faciliter les prédictions.

## Structure du projet

### Notebooks
- `EDA_P7.ipynb` : Analyse exploratoire des données
- `Features_Engi_P7.ipynb` : Ingénierie des features
- `Modeling.ipynb` : Entraînement et évaluation du modèle
- `Data_Drift.ipynb` : Analyse du data drift

### API et Application
- `credit_scoring_api/api/main.py` : API FastAPI pour les prédictions
- `credit_scoring_api/api/app.py` : Interface Streamlit
- `LGBM_TTS.pkl` : Modèle LightGBM entraîné

## Technologies utilisées
- **Machine Learning** : LightGBM, scikit-learn, MLflow
- **API** : FastAPI, Uvicorn
- **Interface** : Streamlit
- **Data** : Pandas, NumPy

## Installation

```bash
pip install -r credit_scoring_api/requirements.txt
```

## Utilisation

### Lancer l'API
```bash
cd credit_scoring_api/api
python main.py
```

### Lancer l'interface Streamlit
```bash
cd credit_scoring_api/api
streamlit run app.py
```

## Fonctionnalités
- Prédiction de scoring crédit par ID client
- Calcul de probabilité de défaut
- Visualisation de l'importance des features
- API REST pour intégration avec d'autres systèmes

## Déploiement
L'API est déployée sur Render : `https://credit-scoring-api-8lkh.onrender.com`

## Données
Le projet utilise les données Home Credit Default Risk avec :
- Données d'applications de crédit
- Historique de bureau de crédit
- Soldes de cartes de crédit
- Historique de paiements
