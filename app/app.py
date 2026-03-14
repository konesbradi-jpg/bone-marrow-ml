<<<<<<< HEAD
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
=======
"""
HémoVision — Interface Streamlit
Design : clinique-moderne, palette navy/cyan/blanc, typographie médicale sobre
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HémoVision — Aide à la Greffe",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL — design médical épuré, palette navy/cyan
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Fond principal ── */
.stApp { background: #f7f9fc; }
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d2146 60%, #0e3060 100%);
    border-right: 1px solid rgba(0,200,255,0.15);
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #00b4d8 !important; }
[data-testid="stSidebar"] label { color: #90caf9 !important; font-size: 0.8rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'DM Serif Display', serif !important;
    border-bottom: 1px solid rgba(0,200,255,0.2);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    color: white !important;
}

/* ── Header principal ── */
.hemovision-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2146 50%, #0a3d6b 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 8px 32px rgba(10,22,40,0.25);
    position: relative;
    overflow: hidden;
}
.hemovision-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hemovision-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important;
    color: white !important;
    margin: 0 0 0.25rem 0 !important;
    line-height: 1.1 !important;
}
.hemovision-header p {
    color: #90caf9 !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    letter-spacing: 0.02em;
}
.header-badge {
    background: rgba(0,180,216,0.2);
    border: 1px solid rgba(0,180,216,0.4);
    color: #00e5ff !important;
    font-size: 0.7rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(10,22,40,0.08);
    border: 1px solid #e8eef5;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #0a1628;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Metric chips ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-chip {
    background: #f0f7ff;
    border: 1px solid #c8dff5;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #0d2146;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-chip strong { display: block; font-size: 1.1rem; color: #0a3d6b; }

/* ── Bouton analyse ── */
.stButton > button {
    background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(0,119,182,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,119,182,0.45) !important;
}

/* ── Résultat succès ── */
.result-success {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 24px rgba(27,94,32,0.3);
    animation: fadeInUp 0.5s ease;
}
/* ── Résultat risque ── */
.result-risk {
    background: linear-gradient(135deg, #7f1d1d, #b71c1c);
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 24px rgba(127,29,29,0.3);
    animation: fadeInUp 0.5s ease;
}
.result-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; margin-bottom: 0.4rem; }
.result-proba { font-size: 2.8rem; font-weight: 600; margin: 0.4rem 0; }
.result-label { font-size: 0.85rem; opacity: 0.85; letter-spacing: 0.05em; }

/* ── Barre de probabilité ── */
.proba-bar-wrap { background: #e8eef5; border-radius: 8px; height: 12px; overflow: hidden; margin: 0.4rem 0 0.2rem; }
.proba-bar-inner { height: 100%; border-radius: 8px; transition: width 1s ease; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f0f4fa !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #4a6080 !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0a1628 !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.1) !important;
}

/* ── Alertes info ── */
.stAlert { border-radius: 10px !important; }

/* ── Section labels sidebar ── */
.sidebar-section-icon {
    font-size: 1.4rem;
    margin-right: 0.4rem;
}

/* ── Animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Divider ── */
hr { border-color: #e0e8f0 !important; margin: 1.2rem 0 !important; }

/* ── Number input / select ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 8px !important;
    border-color: #c8dff5 !important;
}

/* ── SHAP plot background ── */
.element-container iframe, .stImage { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES RESSOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    try:
        model    = joblib.load('models/xgboost_model.pkl')
        features = joblib.load('models/features_list.pkl')
        return model, features
    except FileNotFoundError:
        st.error("❌ Fichiers modèle introuvables dans models/")
        return None, None

model, features_list = load_resources()
if model is None:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hemovision-header">
    <div style="font-size:3.5rem; flex-shrink:0;">🩸</div>
    <div>
        <h1>HémoVision</h1>
        <p>Système d'aide à la décision pour la transplantation pédiatrique de moelle osseuse</p>
        <span class="header-badge">XGBoost · SHAP · Explicable AI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SAISIE DES DONNÉES PATIENT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Dossier Patient")

    # ── Receveur ─────────────────────────────────────────────
    st.markdown("### 👤 Receveur")
    val_p_age    = st.slider("Âge (ans)", 0.0, 20.0, 9.0, 0.5)
    val_p_mass   = st.number_input("Masse corporelle (kg)", 6.0, 104.0, 35.0, 0.5)
    val_p_gender = st.selectbox("Sexe", ["Masculin", "Féminin"])
    val_p_cmv    = st.selectbox("Statut CMV receveur", ["Négatif", "Positif"])
    val_p_abo    = st.selectbox("Groupe ABO receveur", ["O", "A", "B", "AB"])
    val_p_rh     = st.selectbox("Facteur Rh", ["Positif (+)", "Négatif (-)"])
    val_disease  = st.selectbox("Type de maladie", ["ALL", "AML", "Chronique", "Lymphome", "Non-maligne"])
    val_riskgrp  = st.selectbox("Groupe de risque", ["Faible", "Élevé"])
    val_relapse  = st.selectbox("2e greffe après rechute", ["Non", "Oui"])

    # ── Donneur ───────────────────────────────────────────────
    st.markdown("### 🧬 Donneur")
    val_d_age    = st.slider("Âge donneur (ans)", 18.0, 56.0, 33.0, 0.5)
    val_d_abo    = st.selectbox("Groupe ABO donneur", ["O", "A", "B", "AB"])
    val_d_cmv    = st.selectbox("Statut CMV donneur", ["Négatif", "Positif"])
    val_genderm  = st.selectbox("Concordance de genre", ["Autre", "Femme→Homme"])

    # ── Compatibilité ─────────────────────────────────────────
    st.markdown("### 🔬 Compatibilité HLA")
    val_hla      = st.selectbox("Score HLA", ["10/10 (parfait)", "9/10", "8/10", "7/10"])
    val_abo_match = st.selectbox("Compatibilité ABO", ["Compatible", "Incompatible"])
    val_cmvstat  = st.selectbox("Statut CMV croisé", ["0 – les deux négatifs", "1 – donneur–", "2 – receveur–", "3 – les deux positifs"])

    # ── Cellules souches ──────────────────────────────────────
    st.markdown("### 💉 Cellules Souches")
    val_stem_src = st.selectbox("Source", ["Moelle osseuse", "Sang périphérique"])
    val_cd34     = st.number_input("Dose CD34+ (×10⁶/kg)", 0.5, 58.0, 11.0, 0.5)
    val_cd3cd34  = st.number_input("Ratio CD3/CD34", 0.1, 100.0, 5.0, 0.1)
    val_cd3      = st.number_input("Dose CD3+ (×10⁸/kg)", 0.0, 21.0, 4.5, 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DU VECTEUR DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_input() -> pd.DataFrame:
    """
    Construit le DataFrame d'entrée à partir des widgets sidebar.
    Mapping explicite et direct avec les noms exacts des features
    après OHE (issus de features_list.pkl).
    """
    d = {col: 0 for col in features_list}

    # ── Continus ──────────────────────────────────────────────
    if "Recipientage"  in d: d["Recipientage"]  = val_p_age
    if "Rbodymass"     in d: d["Rbodymass"]     = val_p_mass
    if "Donorage"      in d: d["Donorage"]      = val_d_age
    if "CD34kgx10d6"   in d: d["CD34kgx10d6"]   = val_cd34
    if "CD3dCD34"      in d: d["CD3dCD34"]      = val_cd3cd34
    if "CD3dkgx10d8"   in d: d["CD3dkgx10d8"]   = val_cd3

    # ── Receveur binaire ──────────────────────────────────────
    if "Recipientgender_1" in d: d["Recipientgender_1"] = 1 if val_p_gender == "Masculin" else 0
    if "RecipientCMV_1"    in d: d["RecipientCMV_1"]    = 1 if val_p_cmv == "Positif" else 0
    if "RecipientRh_1"     in d: d["RecipientRh_1"]     = 1 if val_p_rh == "Positif (+)" else 0

    abo_map = {"O": None, "A": "1", "B": "2", "AB": "0"}  # OHE drop_first sur base -1
    r_abo = abo_map.get(val_p_abo)
    if r_abo and f"RecipientABO_{r_abo}" in d: d[f"RecipientABO_{r_abo}"] = 1

    disease_map = {"AML": "AML", "Chronique": "chronic", "Lymphome": "lymphoma", "Non-maligne": "nonmalignant"}
    dis = disease_map.get(val_disease)
    if dis and f"Disease_{dis}" in d: d[f"Disease_{dis}"] = 1

    if "Riskgroup_1"    in d: d["Riskgroup_1"]    = 1 if val_riskgrp == "Élevé" else 0
    if "Txpostrelapse_1" in d: d["Txpostrelapse_1"] = 1 if val_relapse == "Oui" else 0

    # ── Donneur binaire ───────────────────────────────────────
    if "DonorCMV_1"  in d: d["DonorCMV_1"]  = 1 if val_d_cmv == "Positif" else 0
    if "Gendermatch_1" in d: d["Gendermatch_1"] = 1 if val_genderm == "Femme→Homme" else 0

    d_abo = abo_map.get(val_d_abo)
    if d_abo and f"DonorABO_{d_abo}" in d: d[f"DonorABO_{d_abo}"] = 1

    # ── Source cellules souches ───────────────────────────────
    if "Stemcellsource_1" in d: d["Stemcellsource_1"] = 1 if val_stem_src == "Sang périphérique" else 0

    # ── Compatibilité HLA ─────────────────────────────────────
    hla_map = {"9/10": "1", "8/10": "2", "7/10": "3"}
    hla = hla_map.get(val_hla.split()[0])
    if hla and f"HLAmatch_{hla}" in d: d[f"HLAmatch_{hla}"] = 1

    if "ABOmatch_1"    in d: d["ABOmatch_1"]    = 1 if val_abo_match == "Compatible" else 0

    cmv_map = {"1 – donneur–": "1", "2 – receveur–": "2", "3 – les deux positifs": "3"}
    cmv = cmv_map.get(val_cmvstat)
    if cmv and f"CMVstatus_{cmv}" in d: d[f"CMVstatus_{cmv}"] = 1

    return pd.DataFrame([d])


# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ PATIENT (colonnes du haut)
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
chips = [
    (c1, "👤 Receveur",   f"{val_p_age:.0f} ans · {val_p_mass:.0f} kg"),
    (c2, "🧬 Donneur",    f"{val_d_age:.0f} ans · {val_d_abo}"),
    (c3, "🔬 HLA",        val_hla.split()[0]),
    (c4, "💉 CD34+",      f"{val_cd34:.1f} ×10⁶/kg"),
    (c5, "🦠 Maladie",    val_disease),
    (c6, "⚠️ Risque",     val_riskgrp),
]
for col, label, val in chips:
    with col:
        st.markdown(f"""
        <div class="metric-chip">
            <div style="font-size:0.72rem;color:#6b87a8;text-transform:uppercase;letter-spacing:.06em">{label}</div>
            <strong>{val}</strong>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1.2rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Analyse Prédictive", "🔍 Explication SHAP", "📘 À propos"])

# ── TAB 1 : PRÉDICTION ────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("""<div class="card">
            <div class="card-title">🩺 Modèle XGBoost</div>
            <p style="color:#4a6080;font-size:0.9rem;line-height:1.6">
            XGBoost (Extreme Gradient Boosting) est entraîné sur les données pré-greffe
            de 187 patients pédiatriques. Il prédit la survie à partir de 44 variables
            cliniques, biologiques et de compatibilité — sans aucune information
            post-opératoire.
            </p>
            <hr/>
            <div style="display:flex;gap:1rem;flex-wrap:wrap">
                <div style="flex:1;min-width:100px;background:#f0f7ff;border-radius:10px;padding:.7rem 1rem;text-align:center">
                    <div style="font-size:0.72rem;color:#6b87a8;text-transform:uppercase">Dataset</div>
                    <strong style="color:#0a3d6b">187 patients</strong>
                </div>
                <div style="flex:1;min-width:100px;background:#f0f7ff;border-radius:10px;padding:.7rem 1rem;text-align:center">
                    <div style="font-size:0.72rem;color:#6b87a8;text-transform:uppercase">Features</div>
                    <strong style="color:#0a3d6b">44 variables</strong>
                </div>
                <div style="flex:1;min-width:100px;background:#f0f7ff;border-radius:10px;padding:.7rem 1rem;text-align:center">
                    <div style="font-size:0.72rem;color:#6b87a8;text-transform:uppercase">Équilibrage</div>
                    <strong style="color:#0a3d6b">SMOTE</strong>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Bouton
        run = st.button("🔬 LANCER L'ANALYSE PRÉDICTIVE")

    with right:
        if run:
            user_data  = build_input()
            prediction = model.predict(user_data)[0]
            proba      = model.predict_proba(user_data)[0]

            # Stockage en session pour SHAP
            st.session_state["user_data"] = user_data
            st.session_state["proba"]     = proba
            st.session_state["pred"]      = prediction

            survive_pct = proba[0] * 100
            risk_pct    = proba[1] * 100

            if prediction == 0:
                st.markdown(f"""
                <div class="result-success">
                    <div class="result-title">✅ Pronostic Favorable</div>
                    <div class="result-proba">{survive_pct:.1f}%</div>
                    <div class="result-label">PROBABILITÉ DE SURVIE ESTIMÉE</div>
                </div>""", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-risk">
                    <div class="result-title">⚠️ Risque Élevé Détecté</div>
                    <div class="result-proba">{risk_pct:.1f}%</div>
                    <div class="result-label">RISQUE DE COMPLICATIONS FATALES</div>
                </div>""", unsafe_allow_html=True)

            # Barres de probabilité
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            bar_col1, bar_col2 = st.columns(2)
            with bar_col1:
                st.markdown(f"""
                <div style="font-size:.78rem;color:#4a6080;margin-bottom:.3rem">Survie</div>
                <div class="proba-bar-wrap">
                  <div class="proba-bar-inner" style="width:{survive_pct:.1f}%;background:linear-gradient(90deg,#2e7d32,#66bb6a)"></div>
                </div>
                <div style="font-size:.82rem;font-weight:600;color:#2e7d32">{survive_pct:.1f}%</div>
                """, unsafe_allow_html=True)
            with bar_col2:
                st.markdown(f"""
                <div style="font-size:.78rem;color:#4a6080;margin-bottom:.3rem">Risque</div>
                <div class="proba-bar-wrap">
                  <div class="proba-bar-inner" style="width:{risk_pct:.1f}%;background:linear-gradient(90deg,#c62828,#ef5350)"></div>
                </div>
                <div style="font-size:.82rem;font-weight:600;color:#c62828">{risk_pct:.1f}%</div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            st.info("ℹ️ Ce résultat est une aide à la décision. Il ne remplace pas le jugement clinique.")

        elif "pred" not in st.session_state:
            st.markdown("""
            <div style="background:#f0f7ff;border:2px dashed #c8dff5;border-radius:14px;
                        padding:2.5rem;text-align:center;color:#6b87a8">
                <div style="font-size:2.5rem;margin-bottom:.5rem">🔬</div>
                <div style="font-weight:600;font-size:1rem;color:#0a3d6b">Prêt pour l'analyse</div>
                <div style="font-size:.85rem;margin-top:.3rem">
                    Remplissez le dossier patient dans le panneau gauche,<br>puis lancez l'analyse.
                </div>
            </div>""", unsafe_allow_html=True)


# ── TAB 2 : SHAP ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""<div class="card" style="margin-bottom:1rem">
        <div class="card-title">🧬 Interprétabilité SHAP</div>
        <p style="color:#4a6080;font-size:.88rem;line-height:1.6;margin:0">
        Les valeurs SHAP (SHapley Additive exPlanations) décomposent la prédiction du modèle
        en contributions individuelles de chaque variable. Une barre vers la droite
        <strong>augmente</strong> le risque prédit ; vers la gauche, elle le <strong>réduit</strong>.
        </p>
    </div>""", unsafe_allow_html=True)

    if "user_data" not in st.session_state:
        st.markdown("""
        <div style="background:#f0f7ff;border:2px dashed #c8dff5;border-radius:14px;
                    padding:2rem;text-align:center;color:#6b87a8">
            <div style="font-size:2rem;margin-bottom:.5rem">💡</div>
            Lancez d'abord une analyse dans l'onglet <strong>Analyse Prédictive</strong>
            pour voir les explications SHAP.
        </div>""", unsafe_allow_html=True)
    else:
        user_data = st.session_state["user_data"]
        try:
            # Extraire l'estimateur XGBoost depuis le Pipeline
            estimator = model
            if hasattr(model, "steps"):
                estimator = model.steps[-1][1]
                # Transformer X avec le scaler du pipeline
                scaler_step = model.steps[0][1]
                user_data_scaled = pd.DataFrame(
                    scaler_step.transform(user_data),
                    columns=user_data.columns
                )
            else:
                user_data_scaled = user_data

            explainer   = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(user_data_scaled)

            # Gestion shape 2D / 3D
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]

            explanation = shap.Explanation(
                values=sv,
                base_values=float(base_val),
                data=user_data_scaled.iloc[0].values,
                feature_names=features_list,
            )

            # Plot SHAP waterfall
            fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
            plt.rcParams.update({
                "font.family": "sans-serif",
                "axes.facecolor": "#f7f9fc",
                "figure.facecolor": "#f7f9fc",
            })
            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.title("Contribution de chaque variable à la prédiction", 
                     fontsize=13, fontweight="bold", color="#0a1628", pad=15)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

            # Top features positives / négatives
            feat_imp = pd.Series(np.abs(sv), index=features_list).sort_values(ascending=False).head(8)
            st.markdown("**Top 8 variables les plus influentes :**")
            cols_shap = st.columns(4)
            for i, (feat, val) in enumerate(feat_imp.items()):
                with cols_shap[i % 4]:
                    st.markdown(f"""
                    <div style="background:white;border:1px solid #e0e8f0;border-radius:10px;
                                padding:.6rem .8rem;margin-bottom:.5rem;font-size:.82rem">
                        <div style="color:#6b87a8;font-size:.72rem;text-transform:uppercase">{feat}</div>
                        <strong style="color:#0a3d6b">{val:.4f}</strong>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur SHAP : {e}")


# ── TAB 3 : À PROPOS ──────────────────────────────────────────────────────────
with tab3:
    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""<div class="card">
            <div class="card-title">🏥 Contexte Médical</div>
            <p style="color:#4a6080;font-size:.88rem;line-height:1.7">
            La transplantation allogénique de cellules souches hématopoïétiques est
            le traitement de référence pour de nombreuses pathologies hématologiques
            pédiatriques graves (LAL, LAM, aplasie médullaire…).
            </p>
            <p style="color:#4a6080;font-size:.88rem;line-height:1.7">
            HémoVision exploite 44 variables pré-opératoires pour estimer la probabilité
            de succès <em>avant</em> la greffe, aidant les équipes médicales à affiner
            leur stratégie thérapeutique.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card">
            <div class="card-title">🔬 Pipeline ML</div>
            <p style="color:#4a6080;font-size:.88rem;line-height:1.7">
            <strong>Données :</strong> 187 patients (BMT dataset, Silesian University)
            <br><strong>Prétraitement :</strong> KNN-Imputation + OHE + SMOTE
            <br><strong>Leakage prevention :</strong> 8 variables post-greffe exclues
            <br><strong>Modèles :</strong> XGBoost · Random Forest · SVM
            <br><strong>Explicabilité :</strong> SHAP TreeExplainer
            </p>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""<div class="card">
            <div class="card-title">⚠️ Avertissement Clinique</div>
            <p style="color:#7f1d1d;font-size:.88rem;line-height:1.7;background:#fff5f5;
                      padding:1rem;border-radius:8px;border-left:3px solid #c62828">
            Cet outil est un <strong>prototype académique</strong> développé dans le cadre
            de la Semaine Coding. Il n'est <strong>pas certifié</strong> pour un usage
            clinique réel. Toute décision médicale doit reposer sur l'évaluation
            d'un professionnel de santé qualifié.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card">
            <div class="card-title">👥 Équipe</div>
            <p style="color:#4a6080;font-size:.88rem;line-height:1.8">
            🎓 <strong>Coding Week — Mars 2026</strong><br>
            Développé par :<br>
            · Jay<br>
            · Léandre Zadi<br>
            · Adama Sana<br>
            · Ilias Janati
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr/>
<p style="text-align:center;color:#9ab0c8;font-size:.78rem;letter-spacing:.05em">
    HémoVision · Coding Week 09–15 Mars 2026 · XGBoost + SHAP · Usage académique uniquement
</p>
""", unsafe_allow_html=True)
>>>>>>> 66debb900212697f87d49a69e294f90032729b57
