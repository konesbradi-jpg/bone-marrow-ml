import streamlit
import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION DE LA PAGE ---
streamlit.set_page_config(
    page_title="HémoPredict | Transplantation Osseuse",
    page_icon="🩺",
    layout="wide"  # Passage en mode large pour mieux répartir les images
)

# --- STYLE CSS PERSONNALISÉ ---
streamlit.markdown("""
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
@streamlit.cache_resource
def charger_modele():
    chemin_modele = 'modele_final.pkl'
    if not os.path.exists(chemin_modele):
        return None
    try:
        return joblib.load(chemin_modele)
    except Exception:
        return None

modele = charger_modele()

# --- 3. BARRE LATÉRALE (SIDEBAR) AVEC IMAGE ---
with streamlit.sidebar:
    streamlit.image("https://img.freepik.com/vecteurs-libre/concept-don-organes-dessine-main_23-2148943806.jpg", use_column_width=True)
    streamlit.title("À propos")
    streamlit.info("""
    **HémoPredict** est un outil d'aide à la décision clinique utilisant l'Intelligence Artificielle
    pour évaluer les chances de succès des greffes pédiatriques.
    """)
    streamlit.divider()
    streamlit.caption("Projet Académique - 2026")

# --- 4. EN-TÊTE PRINCIPAL ---
# On utilise des colonnes pour mettre une image à côté du titre
col_header1, col_header2 = streamlit.columns([1, 3])

with col_header1:
    # Image représentant la recherche médicale / cellules
    streamlit.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=120)

with col_header2:
    streamlit.title("HémoPredict : Analyse de Transplantation")
    streamlit.subheader("Système intelligent de pronostic pédiatrique")

streamlit.divider()

# Message d'erreur si modèle absent
if modele is None:
    streamlit.error("⚠️ Fichier `modele_final.pkl` introuvable. Veuillez placer le modèle dans le répertoire.")
    streamlit.stop()

# --- 5. FORMULAIRE INTERACTIF ---
streamlit.header("📋 Informations Cliniques")

# On crée des "tuiles" avec streamlit.container
with streamlit.container():
    c1, c2, c3 = streamlit.columns(3)
    
    with c1:
        streamlit.write("### 👤 Patient")
        age = streamlit.slider("Âge de l'enfant (années)", 0.0, 18.0, 8.0)
        poids = streamlit.number_input("Poids actuel (kg)", 2.0, 100.0, 25.0)
    
    with c2:
        streamlit.write("### 🧬 Greffon")
        type_donneur = streamlit.selectbox("Origine du donneur", ["Familial", "Non-apparenté", "Sang de cordon"])
        hla_match = streamlit.select_slider("Compatibilité HLA", options=[6, 7, 8, 9, 10], value=10)

    with c3:
        streamlit.write("### 🏥 Contexte")
        streamlit.image("https://img.freepik.com/vecteurs-premium/concept-illustration-medicale-medecin-analysant-donnees-patients-ordinateur_18660-2135.jpg", use_column_width=True)

# --- 6. PRÉDICTION ---
streamlit.markdown(", unsafe_allow_html=True")
if streamlit.button("🚀 CALCULER LE TAUX DE SURVIE ESTIMÉ"):
    
    # Préparation des données (Doit matcher tes colonnes d'entraînement)
    donnees = pd.DataFrame([{'age': age, 'poids': poids, 'type_donneur': type_donneur, 'hla_match': hla_match}])
    
    try:
        prob = modele.predict_proba(donnees)[0][1] * 100
        
        # Affichage du résultat stylisé
        streamlit.markdown("---")
        streamlit.header("🔬 Analyse des résultats")
        
        res_col1, res_col2 = streamlit.columns([2, 1])
        
        with res_col1:
            if prob >= 75:
                streamlit.balloons()
                color = "#28a745"  # Vert
                label = "PRONOSTIC EXCELLENT"
            elif prob >= 50:
                color = "#ffc107"  # Orange
                label = "PRONOSTIC RÉSERVÉ"
            else:
                color = "#dc3545"  # Rouge
                label = "PRONOSTIC CRITIQUE"
            
            # Affichage de la jauge personnalisée en HTML
            streamlit.markdown(f"""
                <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                    <h1 style="color:white; margin:0;">{prob:.1f}%</h1>
                    <b style="color:white;">{label}</b>
                </div>
            """, unsafe_allow_html=True)
            
            streamlit.write("")
            streamlit.progress(int(prob))
            
        with res_col2:
            streamlit.metric(label="Score de Survie", value=f"{prob:.1f}%", delta=f"{prob-50:.1f}% vs Moyenne")
            streamlit.caption("Ce score est basé sur le modèle XGBoost entraîné sur les données historiques.")

    except Exception as e:
        streamlit.error(f"Erreur d'analyse : {e}")

# --- PIED DE PAGE ---
streamlit.markdown(", unsafe_allow_html=True")
streamlit.divider()
streamlit.write("⚕️ *Note : Cet outil est un prototype à but éducatif et ne remplace pas un avis médical.*")