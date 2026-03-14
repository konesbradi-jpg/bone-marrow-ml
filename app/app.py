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

<<<<<<< HEAD
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
=======
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
    val_p_age    = st.number_input("Âge (ans)", 0.0, 20.0, 9.0, 0.01)
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
    val_d_age    = st.number_input("Âge donneur (ans)", 18.0, 56.0, 33.0, 0.01)
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
    (c1, "👤 Receveur",   f"{val_p_age:.2f} ans · {val_p_mass:.1f} kg"),
    (c2, "🧬 Donneur",    f"{val_d_age:.2f} ans · {val_d_abo}"),
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
>>>>>>> b3cdc0d6090005ea0993e910b0317659cf0ec9e3
