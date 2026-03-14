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
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/features.pkl')
        return model, features
    except Exception as e:
        return None, None

model, features_names = load_assets()

# --- 3. STYLE CSS PERSONNALISÉ (AVEC IMAGE D'ARRIÈRE-PLAN) ---
st.markdown(f"""
    <style>
    /* Image d'arrière-plan */
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Superposition pour la lisibilité */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    
    .main {{
        background: rgba(255, 255, 255, 0.85); /* Fond blanc semi-transparent */
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }}

    /* Style des boutons et cartes */
    .stButton>button {{ 
        width: 100%; 
        border-radius: 10px; 
        height: 3.5em; 
        background-color: #2c3e50; 
        color: white; 
        font-weight: bold; 
        border: none;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #34495e;
        transform: translateY(-2px);
    }}
    
    .prediction-box {{ 
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        color: white; 
        margin-bottom: 20px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    }}
    .success-bg {{ background: linear-gradient(135deg, #27ae60, #2ecc71); }}
    .risk-bg {{ background: linear-gradient(135deg, #c0392b, #e74c3c); }}
    
    /* Style des titres pour ressortir sur le fond */
    h1, h2, h3, .stMarkdown {{
        color: #2c3e50;
    }}
    </style>
    """, unsafe_allow_html=True)

# Encapsulation du contenu dans un div principal pour le style
st.markdown('<div class="main">', unsafe_allow_html=True)

# --- 4. LOGIQUE DE PRÉDICTION ---
def make_prediction(input_data):
    df_input = pd.DataFrame(np.nan, index=[0], columns=features_names)
    for key, value in input_data['num'].items():
        if key in features_names:
            df_input[key] = float(value)
    for key, value in input_data['cat'].items():
        dummy_col = f"{key}_{value}"
        related_cols = [c for c in features_names if c.startswith(f"{key}_")]
        df_input[related_cols] = 0
        if dummy_col in features_names:
            df_input[dummy_col] = 1
    prob = model.predict_proba(df_input)[0]
    pred = model.predict(df_input)[0]
    return pred, prob, df_input

# --- 5. INTERFACE UTILISATEUR ---
st.title("🩺 HémoPredict : Système Expert de Transplantation")
st.markdown("---")

if model is None:
    st.error("⚠️ Erreur : Modèle 'best_model.pkl' introuvable. Veuillez entraîner le modèle d'abord.")
    st.stop()

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

st.markdown("---")

if st.button("📊 LANCER L'ANALYSE DU DIAGNOSTIC"):
    data = {
        'num': {'Recipientage': age_p, 'Rbodymass': poids, 'Donorage': age_d, 'Antigen': hla_match, 'CD34kgx10d6': cd34},
        'cat': {'Recipientgender': genre_p, 'Stemcellsource': source, 'ABOmatch': abo_match, 'Riskgroup': risk_group, 'Relapse': relapse, 'RecipientRh': rh_p}
    }

    prediction, probas, df_final = make_prediction(data)
    prob_mort = probas[1] * 100
    prob_survie = probas[0] * 100

    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""<div class="prediction-box risk-bg"><h2>PRONOSTIC : RISQUE ÉLEVÉ</h2><h1>MORTALITÉ : {prob_mort:.1f}%</h1></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="prediction-box success-bg"><h2>PRONOSTIC : SURVIE</h2><h1>CONFIANCE : {prob_survie:.1f}%</h1></div>""", unsafe_allow_html=True)

        fig_prob = px.bar(
            x=['Survie', 'Mortalité'], 
            y=[prob_survie, prob_mort],
            color=['Survie', 'Mortalité'],
            color_discrete_map={'Survie': '#2ecc71', 'Mortalité': '#e74c3c'},
            title="Distribution des Probabilités"
        )
        fig_prob.update_layout(showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_prob, use_container_width=True)

    with res_col2:
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
        fig_gauge.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("🔍 Pourquoi ce diagnostic ?")
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("**Facteurs aggravants détectés :**")
        if hla_match > 2: st.warning(f"⚠️ Mismatch HLA élevé ({hla_match})")
        if relapse == "yes": st.error("⚠️ Antécédents de rechute")
        if risk_group == "high": st.warning("⚠️ Groupe à haut risque")
        if age_p > 50: st.info(f"ℹ️ Âge du patient ({age_p})")

    with exp_col2:
        st.markdown("**Facteurs protecteurs détectés :**")
        if hla_match == 0: st.success("✅ Compatibilité HLA parfaite")
        if age_d < 35: st.success(f"✅ Donneur jeune ({age_d} ans)")
        if cd34 > 5.0: st.success(f"✅ Dose CD34+ robuste")
        if risk_group == "low": st.success("✅ Profil à bas risque")

st.divider()
st.caption("⚕️ Note : Cette IA est un outil d'assistance. Le diagnostic final doit être validé par un hématologue.")
st.markdown('</div>', unsafe_allow_html=True)

