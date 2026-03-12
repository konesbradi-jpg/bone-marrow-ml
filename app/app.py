import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="HémoPredict", layout="wide")

# Liste des colonnes (Noms exacts du dataset .arff)
ALL_COLUMNS = [
    'Stemcellsource', 'ABOmatch', 'RecipientABO', 'survival_time', 
    'Recipientgender', 'Alel', 'HLAgrI', 'Diseasegroup', 'Gendermatch',
    'CD3dkgx10d8', 'IIIV', 'CD34kgx10d6', 'Rbodymass', 'DonorCMV', 
    'CMVstatus', 'HLAmatch', 'Disease', 'HLAmismatch', 'Antigen', 'Relapse',
    'CD3dCD34', 'Recipientageint', 'Recipientage10', 'RecipientCMV', 
    'PLTrecovery', 'Donorage', 'Txpostrelapse', 'Recipientage',
    'time_to_aGvHD_III_IV', 'Riskgroup', 'ANCrecovery', 'extcGvHD', 
    'aGvHDIIIIV', 'Donorage35', 'DonorABO', 'RecipientRh'
]

# --- CHARGEMENT ---
@st.cache_resource
def load_model():
    if os.path.exists('modele_rf.pkl'):
        return joblib.load('modele_rf.pkl')
    return None

model = load_model()

# --- INTERFACE ---
st.title("🏥 HémoPredict")

col1, col2 = st.columns(2)
with col1:
    age_enfant = st.number_input("Âge du receveur", 0, 25, 10)
    genre = st.selectbox("Genre", ["Male", "Female"])
with col2:
    poids_actuel = st.number_input("Poids (kg)", 1.0, 150.0, 30.0)
    type_donneur = st.selectbox("Type de donneur", ["HLA-matched", "HLA-mismatched", "Unrelated"])
    hla_match = st.slider("Nombre de matchs HLA", 0, 10, 10)

# --- LE BOUTON DE CALCUL (VERSION RADICALE) ---
if st.button("🚀 CALCULER LE TAUX DE SURVIE ESTIMÉ"):
    if model is not None:
        # 1. Créer un dictionnaire avec toutes les colonnes d'origine (avant get_dummies)
        # Utilisez les noms EXACTS de votre fichier CSV 'data/bone-marrow.csv'
        donnees_brutes = {
            "age": [age_enfant],
            "poids": [poids_actuel],
            "type_donneur": [type_donneur],
            "hla_match": [hla_match],
            "Recipientgender": [genre] # Assurez-vous que 'genre' est défini par un selectbox
        }
        
        df_entree = pd.DataFrame(donnees_brutes)
        
        # 2. Appliquer get_dummies comme dans train_model.py
        df_transformed = pd.get_dummies(df_entree)
        
        # 3. Synchroniser avec les colonnes du modèle
        # Votre entraînement a généré une liste de colonnes spécifique
        # On charge la liste des colonnes sauvegardées pendant l'entraînement
        try:
            colonnes_entrainement = joblib.load('modele_rf.pkl')
            
            # Créer un DataFrame vide avec les bonnes colonnes
            df_final = pd.DataFrame(0, index=[0], columns=colonnes_entrainement)
            
            # Remplir avec les valeurs transformées qui existent
            for col in df_transformed.columns:
                if col in df_final.columns:
                    df_final[col] = df_transformed[col].values
            
            # 4. Prédiction
            prob = modele.predict_proba(df_final)[0][1] * 100
            
            # Affichage des résultats (votre code actuel avec st.metric, etc.)
            st.success(f"Probabilité de survie : {prob:.1f}%")
            
        except Exception as e:
            st.error(f"Erreur de compatibilité : {e}")