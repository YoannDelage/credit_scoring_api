import streamlit as st
import requests

st.set_page_config(page_title="Prédiction de Scoring Crédit", initial_sidebar_state="collapsed")

# Titre
st.title('Prédiction de Scoring Crédit')

# Message de bienvenue
st.header('Bonjour, veuillez entrer l\'ID du client recherché')

# Input ID du client
sk_id_curr = st.text_input('ID du client', '')

# Fonction pour faire la requête API
def get_prediction(sk_id_curr):
    if not sk_id_curr:
        return "Erreur : L'ID du client ne peut pas être vide."
    
    try:
        sk_id_curr = int(sk_id_curr)  # Convertir l'entrée en entier
    except ValueError:
        return "Erreur : L'ID du client doit être un entier valide."
    
    # URL de l'API (ne changez pas cette URL - elle fonctionne déjà)
    url = 'https://credit-scoring-api-8lkh.onrender.com/predict'
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    payload = {'SK_ID_CURR': sk_id_curr}
    
    try:
        # Afficher les détails de la requête pour débogage
        with st.expander("Détails de la requête"):
            st.write(f"URL: {url}")
            st.write(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        with st.expander("Détails de la réponse"):
            st.write(f"Code de statut: {response.status_code}")
            st.write(f"Réponse brute: {response.text}")
        
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
    with st.spinner('Récupération de la prédiction...'):
        result = get_prediction(sk_id_curr)
    
    if isinstance(result, dict):
        st.success("Prédiction récupérée avec succès !")
        st.write(f"Résultat de la prédiction : {result['resultat']}")
        st.write(f"Prédiction (0 = Crédit accordé, 1 = Crédit refusé) : {result['prediction']}")
    else:
        st.error(result)






