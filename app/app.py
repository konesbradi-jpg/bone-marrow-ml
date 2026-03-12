import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
all_columns = [
    'Stemcellsource', 'ABOmatch', 'RecipientABO', 'survival_time', 
    'Recipientgender', 'Allel', 'HLAgr1', 'Diseasegroup', 'Gendermatch',
    'CD3dkgx10d8', 'IIIV', 'CD34kgx10d6', 'Rbodymass', 'DonorCMV', 
    'CMVstatus', 'HLAmatch', 'Disease', 'HLAmismatch', 'Antigen', 'Relapse',
    'CD3dCD34', 'Recipientageint', 'Recipientage10', 'RecipientCMV', 
    'PLTrecovery', 'Donorage', 'Txpostrelapse', 'Recipientage',
    'time_to_aGvHD_III_IV', 'Riskgroup', 'ANCrecovery', 'extcGvHD', 
    'aGvHDIIIIV', 'Donorage35', 'DonorABO', 'RecipientRh'
]

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="HémoPredict | Transplantation Osseuse",
    page_icon="🩺",
    layout="wide"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    background-color: #007bff;
    color: white;
}
.prediction-card {
    padding: 20px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- 2. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def charger_modele():
    chemin_modele = "modele_rf.pkl"
    if not os.path.exists(chemin_modele):
        return None
    try:
        return joblib.load(chemin_modele)
    except Exception:
        return None

modele = charger_modele()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://img.freepik.com/vecteurs-libre/concept-don-organes-dessine-main_23-2148943806.jpg", use_column_width=True)
    st.title("À propos")
    st.info("""
    **HémoPredict** est un outil d'aide à la décision clinique utilisant l'Intelligence Artificielle
    pour évaluer les chances de succès des greffes pédiatriques.
    """)
    st.divider()
    st.caption("Projet Académique - 2026")

# --- 4. EN-TÊTE ---
col_header1, col_header2 = st.columns([1, 3])

with col_header1:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=120)

with col_header2:
    st.title("HémoPredict : Analyse de Transplantation")
    st.subheader("Système intelligent de pronostic pédiatrique")

st.divider()

# Vérification modèle
if modele is None:
    st.error("⚠️ Fichier `modele_final.pkl` introuvable. Veuillez placer le modèle dans le répertoire.")
    st.stop()

# --- 5. FORMULAIRE ---
st.header("📋 Informations Cliniques")

with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### 👤 Patient")
        age = st.slider("Âge de l'enfant (années)", 0.0, 18.0, 8.0)
        poids = st.number_input("Poids actuel (kg)", 2.0, 100.0, 25.0)

    with c2:
        st.write("### 🧬 Greffon")
        type_donneur = st.selectbox(
            "Origine du donneur",
            ["Familial", "Non-apparenté", "Sang de cordon"]
        )
        hla_match = st.select_slider(
            "Compatibilité HLA",
            options=[6, 7, 8, 9, 10],
            value=10
        )

    with c3:
        st.write("### 🏥 Contexte")
        st.image(
            "https://img.freepik.com/vecteurs-premium/concept-illustration-medicale-medecin-analysant-donnees-patients-ordinateur_18660-2135.jpg",
            use_column_width=True
        )

# --- 6. PRÉDICTION ---
if st.button("Calculer la probabilité"):
    # 1. On crée un dictionnaire avec TOUTES les colonnes
    # On met 0 pour les chiffres et '?' pour le texte (le pipeline gérera le reste)
    default_data = {col: '?' for col in all_columns}
    
    # 2. On remplace par les vraies saisies de l'utilisateur
    default_data['Recipientage'] = age_enfant
    default_data['Rbodymass'] = poids_actuel
    default_data['Recipientgender'] = genre
    default_data['Stemcellsource'] = source_cellules
    # Ajoutez ici les autres variables de votre formulaire...

    # 3. Conversion en DataFrame
    full_input = pd.DataFrame([default_data])

    try:
        # 4. PRÉDICTION
        # Note : Votre pipeline (SimpleImputer) va transformer les '?' en valeurs réelles
        prediction = model.predict(full_input)
        proba = model.predict_proba(full_input)

        if prediction[0] == 1:
            st.success(f"Résultat : Survie prédite avec une probabilité de {proba[0][1]:.2%}")
        else:
            st.error(f"Résultat : Risque élevé avec une probabilité de décès de {proba[0][0]:.2%}")
            
    except Exception as e:
        st.error(f"Erreur technique : {e}")

# --- PIED DE PAGE ---
st.divider()
st.write("⚕️ *Note : Cet outil est un prototype à but éducatif et ne remplace pas un avis médical.*")