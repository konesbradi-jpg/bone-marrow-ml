import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration et Chargement
st.set_page_config(page_title="Hématologie - Aide à la décision", layout="wide")

@st.cache_resource
def load_resources():
    # On charge le meilleur modèle et les colonnes (sans survival_time désormais)
    model = joblib.load('models/best_model.pkl')
    features = joblib.load('models/features.pkl')
    return model, features

try:
    model, features_names = load_resources()
except:
    st.error("⚠️ Erreur : Modèles introuvables. Relancez train_model.py et evaluate_model.py")
    st.stop()

# 2. Identification dynamique des colonnes médicales
def find_col(keyword, exclude=None):
    for c in features_names:
        if keyword.lower() in c.lower():
            if exclude and exclude.lower() in c.lower(): continue
            return c
    return None

# On cherche les colonnes importantes pour les sliders
col_age_p = find_col('age', exclude='donor') or features_names[0]
col_age_d = find_col('donor') or features_names[1]
col_weight = find_col('weight') or features_names[2]

# 3. Interface Utilisateur
st.title("🏥 Prédiction de Succès de Greffe de Moelle")
st.markdown("---")

st.sidebar.header("📋 Saisie des données patient")

def get_user_inputs():
    inputs = {}
    # On crée des sliders pour les variables critiques
    val_age_p = st.sidebar.slider("Âge du Patient", 0, 20, 10)
    val_age_d = st.sidebar.slider("Âge du Donneur", 18, 65, 35)
    val_weight = st.sidebar.number_input("Poids du Patient (kg)", 5.0, 120.0, 50.0)
    
    # On prépare le DataFrame avec TOUTES les colonnes à 0 (valeur neutre par défaut)
    data = {col: [0.0] for col in features_names}
    df = pd.DataFrame(data)
    
    # On injecte les valeurs saisies aux bonnes colonnes
    df[col_age_p] = float(val_age_p)
    df[col_age_d] = float(val_age_d)
    df[col_weight] = float(val_weight)
    
    return df

df_input = get_user_inputs()

# 4. Prédiction et Explication SHAP
col_left, col_right = st.columns([1, 2])

with col_left:
    st.write("### 🩺 Diagnostic")
    if st.button("Lancer l'Analyse"):
        prediction = model.predict(df_input)[0]
        probabilite = model.predict_proba(df_input)[0]

        if prediction == 1:
            st.success(f"**SUCCÈS PRÉDIT** ({probabilite[1]:.1%})")
        else:
            st.error(f"**ÉCHEC PRÉDIT** ({probabilite[0]:.1%})")
            
        st.info("Cette prédiction est basée sur les données cliniques saisies et les tendances historiques.")

with col_right:
    st.write("### 🔍 Explication SHAP (ML Explicable)")
    if st.button("Calculer l'impact des facteurs"):
        # Calcul SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        # Gestion de la structure (RF/XGB)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        if len(sv.shape) == 3: sv = sv[:,:,1]

        # Graphique SHAP Bar Plot pour ce patient
        fig, ax = plt.subplots()
        # On affiche l'importance locale (pour ce patient spécifique)
        importance_locale = np.abs(sv[0])
        indices = np.argsort(importance_locale)[-10:] # Top 10
        
        plt.barh(np.array(features_names)[indices], sv[0][indices], color='skyblue')
        plt.title("Impact des caractéristiques sur cette décision")
        plt.xlabel("Valeur SHAP (Impact)")
        st.pyplot(fig)
        st.caption("Les valeurs positives poussent vers le succès, négatives vers l'échec.")
