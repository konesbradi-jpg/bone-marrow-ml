import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Couleurs pour les graphiques
PALETTE = {"randomforest": "#2196F3", "xgboost": "#FF9800", "svm": "#4CAF50"}

def load_test_data():
    """Charge les données de test sauvegardées lors de l'entraînement."""
    try:
        X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_cleaned.csv"))
        y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test_cleaned.csv")).values.ravel()
        print(f"✅ Données de test chargées : {X_test.shape[0]} patients.")
        return X_test, y_test
    except FileNotFoundError:
        print("❌ Erreur : Fichiers X_test_cleaned.csv ou y_test_cleaned.csv introuvables.")
        return None, None

def get_predictions(model, X_test):
    """Obtient les prédictions et les probabilités, gère le cas particulier du SVM."""
    y_pred = model.predict(X_test)
    
    # Tentative d'obtention des probabilités (pour l'AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Cas SVM sans probability=True : on normalise le score
        decision = model.decision_function(X_test)
        y_prob = (decision - decision.min()) / (decision.max() - decision.min())
    else:
        y_prob = y_pred.astype(float)
        
    return y_pred, y_prob

def run_evaluation():
    print("\n" + "="*50)
    print("🔬 ÉVALUATION COMPARATIVE DES MODÈLES")
    print("="*50)

    X_test, y_test = load_test_data()
    if X_test is None: return

    results = []
    roc_curves = {}
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.pkl')]
    
    if not model_files:
        print("❌ Aucun modèle trouvé dans le dossier 'models/'.")
        return

    fig_cm, axes_cm = plt.subplots(1, len(model_files), figsize=(5*len(model_files), 4))
    if len(model_files) == 1: axes_cm = [axes_cm]

    for i, file in enumerate(model_files):
        name = file.replace('_model.pkl', '')
        model = joblib.load(os.path.join(MODELS_DIR, file))
        
        y_pred, y_prob = get_predictions(model, X_test)
        
        # Calcul des métriques
        metrics = {
            "Modèle": name.upper(),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)
        
        # Stockage pour courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, metrics["AUC"])

        # Matrice de Confusion
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Survie", "Mort"])
        disp.plot(ax=axes_cm[i], cmap="Blues", colorbar=False)
        axes_cm[i].set_title(f"CM: {name.upper()}")

    # 1. Affichage des résultats
    df_results = pd.DataFrame(results).set_index("Modèle")
    print("\n📊 RÉSULTATS COMPARATIFS :")
    print(df_results.sort_values(by="AUC", ascending=False))
    df_results.to_csv(os.path.join(OUTPUT_DIR, "comparaison_modeles.csv"))

    # 2. Sauvegarde des Matrices de Confusion
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matrices_confusion.png"))
    plt.close()

    # 3. Courbes ROC
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Comparaison des Courbes ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "courbes_roc.png"))
    plt.close()

    # 4. Identification du meilleur modèle et SHAP
    best_model_name = df_results["AUC"].idxmax().lower()
    print(f"\n🏆 Meilleur modèle détecté : {best_model_name.upper()}")
    
    best_pipe = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name}_model.pkl"))
    
    # Sauvegarde comme modèle de production
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_model.pkl"))

    print(f"\n🧠 Génération de l'analyse SHAP pour {best_model_name.upper()}...")
    try:
        # On extrait le classifieur et on transforme les données via le pipeline (imputer/scaler)
        # On suppose que le pipeline a une étape 'clf' à la fin
        clf = best_pipe.named_steps['clf']
        preprocessor = best_pipe[:-1] # Tout sauf le dernier élément
        X_test_transformed = preprocessor.transform(X_test)
        
        # On utilise KernelExplainer pour SVM ou TreeExplainer pour RF/XGB
        if best_model_name == "svm":
            explainer = shap.KernelExplainer(clf.predict_proba, shap.sample(X_test_transformed, 50))
        else:
            explainer = shap.TreeExplainer(clf)
            
        shap_values = explainer.shap_values(X_test_transformed)
        
        # Plot SHAP
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list): # Cas RandomForest (une liste par classe)
            shap.summary_plot(shap_values[1], X_test_transformed, feature_names=X_test.columns, show=False)
        else: # Cas XGBoost
            shap.summary_plot(shap_values, X_test_transformed, feature_names=X_test.columns, show=False)
            
        plt.title(f"Importance des variables (SHAP) - {best_model_name.upper()}")
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), bbox_inches='tight')
        print(f"✅ Analyse SHAP sauvegardée dans {OUTPUT_DIR}/shap_importance.png")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'analyse SHAP : {e}")

    print("\n✅ Évaluation terminée. Consultez le dossier 'outputs/' pour les graphiques.")

if __name__ == "__main__":
    run_evaluation()