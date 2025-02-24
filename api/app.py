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
        try:
            sk_id_curr = int(sk_id_curr)  # Essayer de convertir l'entrée en entier
        except ValueError:
            return "Erreur : L'ID du client doit être un entier valide."
        
        url = 'https://credit-scoring-api-yd.herokuapp.com/predict' 
        payload = {'SK_ID_CURR': sk_id_curr}  # on passe les données sous forme de dictionnaire
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
        if isinstance(result, dict):  # Si la réponse est un dictionnaire (réponse API)
            st.write(f"Résultat de la prédiction : {result['resultat']}")
            st.write(f"Prédiction (0 = Crédit accordé, 1 = Crédit refusé) : {result['prediction']}")
        else:  # Si ce n'est pas un dictionnaire, afficher l'erreur
            st.write(result)
    else:
        st.write("Erreur dans la prédiction.")


