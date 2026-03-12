import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Liste exhaustive des colonnes attendues par le pipeline
all_columns = [
    'Stemcellsource', 'ABOmatch', 'RecipientABO', 'survival_time', 
    'Recipientgender', 'Alel', 'HLAgrI', 'Diseasegroup', 'Gendermatch',
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

# --- STYLE CSS ---
st.markdown("""
<style>
.stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def charger_modele():
    # Assure-toi que ce nom correspond au fichier généré par ton script d'entraînement
    chemin_modele = "modele_rf.pkl" 
    if not os.path.exists(chemin_modele):
        return None
    try:
        return joblib.load(chemin_modele)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

modele = charger_modele()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("💡 Aide")
    st.info("Remplissez les informations cliniques à droite pour obtenir un pronostic de survie.")
    st.divider()
    st.caption("Version 1.0 - HémoPredict")

# --- 4. EN-TÊTE ---
st.title("🩺 HémoPredict : Analyse de Transplantation")
st.subheader("Système intelligent de pronostic pédiatrique")
st.divider()

if modele is None:
    st.error("⚠️ Fichier `modele_final.pkl` introuvable. Veuillez placer le modèle dans le dossier du projet.")
    st.stop()

# --- 5. FORMULAIRE (Saisie utilisateur) ---
st.header("📋 Informations Cliniques")

with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("### 👤 Patient")
        age = st.slider("Âge du patient (années)", 0.0, 20.0, 10.0)
        poids = st.number_input("Poids (kg)", 2.0, 150.0, 45.0)
        genre = st.selectbox("Genre du patient", ["F", "M"])

    with c2:
        st.write("### 🧬 Greffon")
        source_cellules = st.selectbox(
            "Source des cellules souches",
            ["peripheral_blood", "bone_marrow", "cord_blood"]
        )
        hla_match = st.select_slider(
            "Compatibilité HLA (Antigen)",
            options=[0, 1, 2, 3], # Selon l'encodage du dataset
            value=0
        )

    with c3:
        st.write("### 🏥 Risques")
        groupe_risque = st.selectbox("Groupe de Risque", ["High", "Low"])
        cmv_status = st.selectbox("Statut CMV", [0, 1, 2, 3])

# --- 6. LOGIQUE DE PRÉDICTION ---
if st.button("Lancer l'Analyse"):
    # 1. Créer un dictionnaire avec TOUTES les colonnes initialisées à NaN
    # Le SimpleImputer du pipeline remplira ces valeurs manquantes automatiquement
    entree_data = {col: np.nan for col in all_columns}
    
    # 2. Remplir avec les variables saisies par l'utilisateur
    # Attention : les noms ici doivent être EXACTEMENT ceux de 'all_columns'
    entree_data['Recipientage'] = age
    entree_data['Rbodymass'] = poids
    entree_data['Recipientgender'] = genre
    entree_data['Stemcellsource'] = source_cellules
    entree_data['Antigen'] = hla_match
    entree_data['Riskgroup'] = groupe_risque
    entree_data['CMVstatus'] = cmv_status

    # 3. Conversion en DataFrame (avec l'ordre exact des colonnes)
    full_input = pd.DataFrame([entree_data])[all_columns]

    try:
        # 4. Prédiction
        prediction = modele.predict(full_input)
        proba = modele.predict_proba(full_input)

        st.divider()
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if prediction[0] == 1:
                st.success("### ✅ PRONOSTIC : SURVIE")
                st.metric("Confiance", f"{proba[0][1]:.2%}")
            else:
                st.error("### ⚠️ PRONOSTIC : RISQUE ÉLEVÉ")
                st.metric("Confiance", f"{proba[0][0]:.2%}")

        with col_res2:
            # Petit graphique de probabilité
            prob_df = pd.DataFrame({
                'Issue': ['Décès', 'Survie'],
                'Probabilité': proba[0]
            })
            st.bar_chart(prob_df.set_index('Issue'))

    except Exception as e:
        st.error(f"❌ Une erreur est survenue lors de la prédiction : {e}")
        st.info("Astuce : Vérifiez que le pipeline de votre modèle inclut bien un 'Imputer' pour gérer les colonnes vides.")

# --- PIED DE PAGE ---
st.divider()
st.caption("⚕️ Note : Cet outil est un prototype à but éducatif. Les décisions médicales doivent être prises par des professionnels.")