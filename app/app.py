import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from imblearn.pipeline import Pipeline # CRUCIAL pour charger le modèle

# Configuration des chemins (doit être identique à train.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
INFO_PATH = os.path.join(BASE_DIR, "models", "features_info.pkl")

st.set_page_config(page_title="HémoPredict Expert", layout="wide")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        info = joblib.load(INFO_PATH)
        return model, info
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None

model, info = load_assets()

# --- Interface utilisateur ---
st.title("🧬 HémoPredict Expert Analysis")

if model is None:
    st.error(f"⚠️ Le modèle est introuvable !")
    st.info(f"Veuillez vérifier que le fichier existe ici : `{MODEL_PATH}`")
    st.warning("Action : Lancez d'abord le script `train.py` dans votre terminal.")
    st.stop()

# --- Formulaire ---
with st.container():
    tabs = st.tabs(["👤 Patient", "🧬 Donneur", "🏥 Clinique"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        Recipientage = c1.number_input("Âge Patient", 0, 80, 15)
        Rbodymass = c2.number_input("Poids (kg)", 5, 150, 50)
        Recipientgender = c1.selectbox("Genre", ["M", "F"])
        RecipientRh = c2.selectbox("Rhésus Patient", ["plus", "minus"])
    with tabs[1]:
        c1, c2 = st.columns(2)
        Donorage = c1.number_input("Âge Donneur", 18, 80, 30)
        Stemcellsource = c2.selectbox("Source", ["bone_marrow", "peripheral_blood"])
        ABOmatch = c1.selectbox("ABO Match", ["full_match", "minor_mismatch", "major_mismatch", "bidirectional_mismatch"])
    with tabs[2]:
        c1, c2 = st.columns(2)
        Alel = c1.number_input("Alel (Mismatch)", 0, 10, 0)
        Antigen = c2.number_input("Antigen (Mismatch)", 0, 10, 0)
        Riskgroup = c1.selectbox("Risk Group", ["low", "high"])
        CMVstatus = c2.selectbox("CMV Status", ["0", "1", "2", "3"])
        Relapse = c1.selectbox("Relapse", ["no", "yes"])
        CD34kgx10d6 = c2.number_input("CD34+ (10^6/kg)", 0.0, 50.0, 5.0)

if st.button("🔍 CALCULER LE PRONOSTIC", use_container_width=True):
    # Préparation des données dans le format exact attendu par le pipeline
    input_df = pd.DataFrame([{
        'Recipientage': Recipientage, 'Rbodymass': Rbodymass, 'Recipientgender': Recipientgender,
        'RecipientRh': RecipientRh, 'Donorage': Donorage, 'Stemcellsource': Stemcellsource,
        'ABOmatch': ABOmatch, 'Alel': Alel, 'Antigen': Antigen, 'Riskgroup': Riskgroup,
        'CMVstatus': CMVstatus, 'Relapse': Relapse, 'CD34kgx10d6': CD34kgx10d6
    }])
    
    # Réorganiser les colonnes
    input_df = input_df[info['features']]
    
    prob = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]
    
    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        color = "#10b981" if pred == 0 else "#ef4444"
        label = "SURVIE" if pred == 0 else "RISQUE"
        st.markdown(f"""<div style="background:{color}; padding:2rem; border-radius:15px; text-align:center; color:white;">
            <h2>PRONOSTIC : {label}</h2><h1>{prob[pred]*100:.1f}%</h1></div>""", unsafe_allow_html=True)
    with res2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob[1] * 100,
            title = {'text': "Risque de Mortalité (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1e293b"}}))
        st.plotly_chart(fig, use_container_width=True)