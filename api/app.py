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
    if not sk_id_curr:
        return "Erreur : L'ID du client ne peut pas être vide."

    try:
        sk_id_curr = int(sk_id_curr)  # Essayer de convertir l'entrée en entier
    except ValueError:
        return "Erreur : L'ID du client doit être un entier valide."
    
    url = 'https://credit-scoring-api-yd-268c4aa564a3.herokuapp.com/predict'
    payload = {'SK_ID_CURR': sk_id_curr}  # on passe les données sous forme de dictionnaire

    try:
        response = requests.post(url, json=payload)  # Utilisation de POST
        
        # Si le code de réponse n'est pas 200
        if response.status_code != 200:
            return f"Erreur dans la prédiction : Code de réponse {response.status_code} - {response.text}"
        
        # Si la réponse est correcte, on la traite
        prediction = response.json()
        
        # Vérifier si la réponse contient les clés attendues
        if 'resultat' not in prediction or 'prediction' not in prediction:
            return "Erreur : La réponse de l'API est mal formée."

        return prediction
    
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion à l'API : {str(e)}"
    except Exception as e:
        return f"Erreur inconnue : {str(e)}"

# ajout bouton
if st.button('Obtenir la prédiction'):
    result = get_prediction(sk_id_curr)
    if isinstance(result, dict):  # Si la réponse est un dictionnaire (réponse API)
        st.write(f"Résultat de la prédiction : {result['resultat']}")
        st.write(f"Prédiction (0 = Crédit accordé, 1 = Crédit refusé) : {result['prediction']}")
    else:  # Si ce n'est pas un dictionnaire, afficher l'erreur
        st.write(result)



