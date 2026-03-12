import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="HémoPredict | Analyse de Transplantation",
    page_icon="🩺",
    layout="wide"
)

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/features.pkl')
        return model, features
    except Exception as e:
        return None, None

model, features_names = load_assets()

# --- 3. STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #0056b3; border: none; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px; }
    .success-bg { background: linear-gradient(135deg, #28a745, #218838); }
    .risk-bg { background: linear-gradient(135deg, #dc3545, #c82333); }
    </style>
""", unsafe_allow_html=True)

# --- 4. LOGIQUE DE PRÉDICTION ---
def make_prediction(input_data):
    # Créer un DataFrame vide avec toutes les colonnes du modèle (initialisé à 0)
    df_input = pd.DataFrame(0, index=[0], columns=features_names)
    
    # Remplir les colonnes numériques
    for key, value in input_data['num'].items():
        # Trouver la colonne exacte dans features_names qui contient le nom
        for col in features_names:
            if key.lower() in col.lower() and "_" not in col: # Colonne numérique pure
                df_input[col] = value

    # Remplir les colonnes catégorielles (Dummies)
    for key, value in input_data['cat'].items():
        # Reconstruire le nom de la colonne dummy : "NomVariable_Valeur"
        dummy_col = f"{key}_{value}"
        if dummy_col in features_names:
            df_input[dummy_col] = 1

    # Prédire
    prob = model.predict_proba(df_input)[0]
    pred = model.predict(df_input)[0]
    return pred, prob

# --- 5. INTERFACE UTILISATEUR ---
if model is None:
    st.error("⚠️ Fichiers de modèle introuvables. Veuillez d'abord exécuter train_model.py")
    st.stop()

st.title("🩺 HémoPredict : Système Expert de Transplantation")
st.markdown("---")

# Création des colonnes de saisie
col_p, col_d, col_m = st.columns(3)

with col_p:
    st.subheader("👤 Patient (Receveur)")
    age_p = st.number_input("Âge du Patient", 0, 80, 25)
    poids = st.number_input("Poids (kg)", 5.0, 150.0, 65.0)
    genre_p = st.selectbox("Genre Patient", ["M", "F"], key="gp")
    rh_p = st.selectbox("Rhésus Patient", ["plus", "minus"], key="rhp")
    cmv_p = st.selectbox("Statut CMV Patient", [0, 1], key="cmvp")

with col_d:
    st.subheader("🧬 Donneur & Greffon")
    age_d = st.number_input("Âge du Donneur", 18, 80, 30)
    source = st.selectbox("Source des Cellules", ["bone_marrow", "peripheral_blood", "cord_blood"])
    abo_match = st.selectbox("Match ABO", ["full_match", "minor_mismatch", "major_mismatch", "bidirectional_mismatch"])
    donor_cmv = st.selectbox("Statut CMV Donneur", [0, 1])
    gendermatch = st.selectbox("Match de Genre", ["other", "female_to_male"])

with col_m:
    st.subheader("🚩 Facteurs Cliniques")
    risk_group = st.selectbox("Groupe de Risque", ["low", "high"])
    disease_group = st.selectbox("Groupe de Maladie", ["nonsmalignant", "malignant"])
    relapse = st.selectbox("Antécédent de Rechute", ["no", "yes"])
    hla_match = st.slider("Nombre d'antigènes HLA mismatch", 0, 10, 0)
    cd34 = st.number_input("Dose CD34+ (10^6/kg)", 0.0, 50.0, 5.0)

# --- 6. ACTION ET AFFICHAGE DES RÉSULTATS ---
st.markdown("---")
if st.button("LANCER L'ANALYSE PRONOSTIQUE"):
    # Structuration des données
    data = {
        'num': {
            'Recipientage': age_p,
            'Rbodymass': poids,
            'Donorage': age_d,
            'Antigen': hla_match,
            'CD34kgx10d6': cd34
        },
        'cat': {
            'Recipientgender': genre_p,
            'Stemcellsource': source,
            'ABOmatch': abo_match,
            'Riskgroup': risk_group,
            'Diseasegroup': disease_group,
            'Relapse': relapse,
            'RecipientRh': rh_p,
            'Gendermatch': gendermatch
        }
    }

    prediction, probabilites = make_prediction(data)
    
    # Zone de résultat
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        score_survie = probabilites[1] * 100
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box success-bg">
                    <h2>PRONOSTIC : SURVIE</h2>
                    <h1 style="font-size: 4em;">{score_survie:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box risk-bg">
                    <h2>PRONOSTIC : RISQUE ÉLEVÉ</h2>
                    <h1 style="font-size: 4em;">{probabilites[0]*100:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Jauge Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_survie,
            title = {'text': "Indice de Confiance Survie"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#007bff"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#d4edda"}
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- Section Explicative ---
    with st.expander("ℹ️ Analyse détaillée des facteurs"):
        st.write("Voici l'impact des variables clés pour ce patient :")
        impact_cols = st.columns(3)
        with impact_cols[0]:
            st.metric("Incompatibilité HLA", f"{hla_match} Ag", delta="Risque" if hla_match > 0 else "Optimal", delta_color="inverse")
        with impact_cols[1]:
            st.metric("Dose CD34+", f"{cd34}", delta="Normal" if cd34 > 2 else "Faible")
        with impact_cols[2]:
            st.metric("Âge Donneur", f"{age_d} ans", delta="Jeune" if age_d < 35 else "Elevé", delta_color="inverse")

st.divider()
st.caption("⚕️ Cet outil est une aide à la décision clinique basée sur des données historiques. La décision finale appartient à l'équipe médicale.")