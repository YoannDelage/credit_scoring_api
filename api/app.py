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
        url = 'https://credit-scoring-api-yd.herokuapp.com/predict' 
        payload = {'SK_ID_CURR': int(sk_id_curr)}  # on passe les données sous forme de dictionnaire
        response = requests.post(url, json=payload)  # Utilisation de POST
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
        st.write(f"Résultat de la prédiction : {result['resultat']}")
        st.write(f"Prédiction (0 = Crédit accordé, 1 = Crédit refusé) : {result['prediction']}")
    else:
        st.write("Erreur dans la prédiction.")

