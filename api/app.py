import streamlit as st
import requests
import json

# titre
st.title('Prédiction de Scoring Crédit')

# message de bienvenue
st.header('Bonjour, veuillez entrer l\'ID du client recherché')

# input ID du client
sk_id_curr = st.text_input('ID du client', '')

# fonction pour faire la requete API
def get_prediction(sk_id_curr):
    if sk_id_curr:
        url = f'https://git.heroku.com/credit-scoring-api-yd.git/predict/{sk_id_curr}' 
        response = requests.get(url)
        if response.status_code == 200:
            prediction = response.json()
            return prediction
        else:
            return "Erreur dans la prédiction"
    return None

# ajout bouton
if st.button('Obtenir la prédiction'):
    result = get_prediction(sk_id_curr)
    if result:
        st.write(f"Résultat de la prédiction : {json.dumps(result, indent=2)}")
    else:
        st.write("Veuillez entrer un ID de client valide.")
