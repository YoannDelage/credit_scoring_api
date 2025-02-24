import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

# chgt du modele MLflow depuis registry
def load_model(model_name: str):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}")
    return model

# fonction de prediction
def predict(model, features):
    # veriif si features deja dans un DF
    if not isinstance(features, pd.DataFrame):
        # si format = np.array, convert en DF
        df = pd.DataFrame(features)
    else:
        df = features
        
    # pred
    prediction = model.predict(df)
    
    # verif que prediction est un scalaire
    if hasattr(prediction, "__len__") and len(prediction) == 1:
        return prediction[0]
    return prediction