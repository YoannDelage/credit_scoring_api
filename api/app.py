import streamlit as st
import requests
import json

# Désactivation de la protection XSRF
st.set_page_config(page_title="Prédiction de Scoring Crédit", initial_sidebar_state="collapsed")

# Titre
st.title('Prédiction de Scoring Crédit')

# Message de bienvenue
st.header('Bonjour, veuillez entrer l\'ID du client recherché')

# Input ID du client
sk_id_curr = st.text_input('ID du client', '')

# Clé d'API
API_KEY = 'HRKU-b32a3477-3def-45ff-af77-23caa503e3fc'

# Fonction pour faire la requête API
def get_prediction(sk_id_curr):
    if not sk_id_curr:
        return "Erreur : L'ID du client ne peut pas être vide."
    
    try:
        sk_id_curr = int(sk_id_curr)  # Convertir l'entrée en entier
    except ValueError:
        return "Erreur : L'ID du client doit être un entier valide."
    
    url = 'https://credit-scoring-api-yd-268c4aa564a3.herokuapp.com/predict'
    headers = {
        'api_key': API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {'SK_ID_CURR': sk_id_curr}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return f"Erreur dans la prédiction : Code {response.status_code} - {response.text}"
        
        prediction = response.json()
        
        if 'resultat' not in prediction or 'prediction' not in prediction:
            return "Erreur : La réponse de l'API est mal formée."
        
        return prediction
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion à l'API : {str(e)}"
    except Exception as e:
        return f"Erreur inconnue : {str(e)}"

# Ajout bouton
if st.button('Obtenir la prédiction'):
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = get_prediction(sk_id_curr)
    
    result = st.session_state.prediction_result
    
    if isinstance(result, dict):
        st.write(f"Résultat de la prédiction : {result['resultat']}")
        st.write(f"Prédiction (0 = Crédit accordé, 1 = Crédit refusé) : {result['prediction']}")
    else:
        st.write(result)






