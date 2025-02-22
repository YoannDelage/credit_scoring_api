import mlflow
import mlflow.pyfunc
import pandas as pd

# Charger ton modèle MLflow depuis le model registry ou le modèle sauvegardé
def load_model(model_name: str):
    # Si ton modèle est dans un "model registry" MLflow
    model = mlflow.pyfunc.load_model(f"models:/LightGBM_smoted_tuned_trained.pkl/3")
    return model

# Fonction de prédiction
def predict(model, features: list):
    # Convertir les données en DataFrame
    df = pd.DataFrame([features])

    # Effectuer la prédiction
    prediction = model.predict(df)
    return prediction