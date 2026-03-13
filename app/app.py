import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="HémoPredict | Diagnostic Expert",
    page_icon="🩺",
    layout="wide"
)

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    try:
        # On charge le Pipeline (Imputer + Scaler + Modèle)
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/features.pkl')
        return model, features
    except Exception as e:
        return None, None

model, features_names = load_assets()

# --- 3. STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #2c3e50; color: white; font-weight: bold; border: none; }
    .prediction-box { padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .success-bg { background: linear-gradient(135deg, #27ae60, #2ecc71); } /* Vert pour Survie */
    .risk-bg { background: linear-gradient(135deg, #c0392b, #e74c3c); }    /* Rouge pour Mort */
    .reason-card { background-color: white; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- 4. LOGIQUE DE PRÉDICTION ---
def make_prediction(input_data):
    # Création du DataFrame avec NaN (le KNNImputer gérera les vides)
    df_input = pd.DataFrame(np.nan, index=[0], columns=features_names)
    
    # Remplissage des numériques
    for key, value in input_data['num'].items():
        if key in features_names:
            df_input[key] = float(value)

    # Remplissage des catégoriels (Dummies)
    for key, value in input_data['cat'].items():
        dummy_col = f"{key}_{value}"
        # On réinitialise les colonnes liées à cette catégorie
        related_cols = [c for c in features_names if c.startswith(f"{key}_")]
        df_input[related_cols] = 0
        if dummy_col in features_names:
            df_input[dummy_col] = 1

    # Prédiction via Pipeline
    prob = model.predict_proba(df_input)[0]
    pred = model.predict(df_input)[0]
    return pred, prob, df_input

# --- 5. INTERFACE UTILISATEUR ---
st.title("🩺 HémoPredict : Système Expert de Transplantation")
st.markdown("---")

if model is None:
    st.error("⚠️ Erreur : Modèle 'best_model.pkl' introuvable. Veuillez entraîner le modèle d'abord.")
    st.stop()

# FORMULAIRE EN 3 COLONNES
col_p, col_d, col_m = st.columns(3)

with col_p:
    st.subheader("👤 Profil Patient")
    age_p = st.number_input("Âge du Patient", 0, 80, 25)
    poids = st.number_input("Poids (kg)", 5.0, 150.0, 65.0)
    genre_p = st.selectbox("Genre Patient", ["M", "F"])
    rh_p = st.selectbox("Rhésus Patient", ["plus", "minus"])

with col_d:
    st.subheader("🧬 Donneur & Greffon")
    age_d = st.number_input("Âge du Donneur", 18, 80, 30)
    source = st.selectbox("Source", ["bone_marrow", "peripheral_blood", "cord_blood"])
    abo_match = st.selectbox("Match ABO", ["full_match", "minor_mismatch", "major_mismatch", "bidirectional_mismatch"])

with col_m:
    st.subheader("🚩 Facteurs de Risque")
    risk_group = st.selectbox("Groupe de Risque", ["low", "high"])
    relapse = st.selectbox("Antécédent de Rechute", ["no", "yes"])
    hla_match = st.slider("HLA mismatch (Antigen)", 0, 10, 0)
    cd34 = st.number_input("Dose CD34+ (10^6/kg)", 0.0, 50.0, 5.0)

# --- 6. ACTION ET AFFICHAGE ---
st.markdown("---")
if st.button("📊 LANCER L'ANALYSE DU DIAGNOSTIC"):
    data = {
        'num': {'Recipientage': age_p, 'Rbodymass': poids, 'Donorage': age_d, 'Antigen': hla_match, 'CD34kgx10d6': cd34},
        'cat': {'Recipientgender': genre_p, 'Stemcellsource': source, 'ABOmatch': abo_match, 'Riskgroup': risk_group, 'Relapse': relapse, 'RecipientRh': rh_p}
    }

    prediction, probas, df_final = make_prediction(data)
    
    # probas[0] = Survie, probas[1] = Mort (Cible 1)
    prob_mort = probas[1] * 100
    prob_survie = probas[0] * 100

    # RÉSULTAT PRINCIPAL
    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""<div class="prediction-box risk-bg"><h2>PRONOSTIC : RISQUE ÉLEVÉ</h2><h1>MORTALITÉ : {prob_mort:.1f}%</h1></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="prediction-box success-bg"><h2>PRONOSTIC : SURVIE</h2><h1>CONFIANCE : {prob_survie:.1f}%</h1></div>""", unsafe_allow_html=True)

        # Graphique à barres des probabilités
        fig_prob = px.bar(
            x=['Survie', 'Mortalité'], 
            y=[prob_survie, prob_mort],
            color=['Survie', 'Mortalité'],
            color_discrete_map={'Survie': '#2ecc71', 'Mortalité': '#e74c3c'},
            title="Distribution des Probabilités"
        )
        fig_prob.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_prob, use_container_width=True)

    with res_col2:
        # Jauge de Risque
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_mort,
            title = {'text': "Indice de Risque Vital (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#34495e"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 60], 'color': "#fff3cd"},
                    {'range': [60, 100], 'color': "#f8d7da"}
                ],
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- 7. RAISONS DU DIAGNOSTIC (Interprétabilité) ---
    st.subheader("🔍 Pourquoi ce diagnostic ?")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("**Facteurs aggravants détectés :**")
        if hla_match > 2:
            st.warning(f"⚠️ Mismatch HLA élevé ({hla_match}) : Augmente fortement le risque de rejet.")
        if relapse == "yes":
            st.error("⚠️ Antécédents de rechute : Facteur historique de mortalité élevé.")
        if risk_group == "high":
            st.warning("⚠️ Groupe à haut risque : Profil clinique complexe.")
        if age_p > 50:
            st.info(f"ℹ️ Âge du patient ({age_p}) : Facteur de fragilité physiologique.")
        if not (hla_match > 2 or relapse == "yes" or risk_group == "high"):
            st.write("Aucun facteur critique majeur détecté.")

    with exp_col2:
        st.markdown("**Facteurs protecteurs détectés :**")
        if hla_match == 0:
            st.success("✅ Compatibilité HLA parfaite : Facteur n°1 de survie.")
        if age_d < 35:
            st.success(f"✅ Donneur jeune ({age_d} ans) : Meilleure qualité du greffon.")
        if cd34 > 5.0:
            st.success(f"✅ Dose CD34+ robuste ({cd34}) : Favorise la prise de greffe.")
        if risk_group == "low":
            st.success("✅ Profil à bas risque : Historique favorable.")

st.divider()
st.caption("⚕️ Note : Cette IA est un outil d'assistance. Le diagnostic final doit être validé par un hématologue.")