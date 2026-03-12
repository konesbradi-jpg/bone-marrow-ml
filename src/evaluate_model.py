import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, average_precision_score, f1_score
)
# Import personnalisé (assure-toi que le chemin est correct)
from src.data_processing import load_and_clean_data

def evaluate_medical_model(model, X, y, model_name="Modèle"):
    """
    Fonction robuste d'évaluation de performance avec métriques médicales.
    """
    # 1. Prédictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # 2. Calcul des métriques de haut niveau
    roc_auc = auc(*roc_curve(y, y_prob)[:2]) if y_prob is not None else 0
    avg_precision = average_precision_score(y, y_prob) if y_prob is not None else 0
    
    print(f"\n" + "="*40)
    print(f"RAPPORT D'ÉVALUATION : {model_name}")
    print("="*40)
    print(classification_report(y, y_pred, target_names=['Décès (0)', 'Survie (1)']))
    print(f" Score AUC-ROC : {roc_auc:.4f}")
    print(f" Average Precision : {avg_precision:.4f}")

    # 3. VISUALISATION
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- A. Matrice de Confusion (Normalisée pour voir les % d'erreurs) ---
    cm = confusion_matrix(y, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Matrice de Confusion (Rappel %)')
    axes[0].set_xlabel('Prédictions')
    axes[0].set_ylabel('Réalité')

    # --- B. Courbe ROC ---
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_title('Courbe ROC (Sensibilité vs Spécificité)')
        axes[1].legend(loc="lower right")

    # --- C. Importance des variables (Si disponible) ---
    # Très utile pour le Gradient Boosting (modele_gb.pkl)
    try:
        # On essaie de récupérer l'importance des variables du classifieur à l'intérieur du pipeline
        if hasattr(model, 'named_steps'):
            clf = model.named_steps['classifier']
            # On récupère les noms de colonnes après transformation si possible
            # Ici on simplifie en prenant les colonnes de X
            importances = pd.Series(clf.feature_importances_, index=X.columns)
        else:
            importances = pd.Series(model.feature_importances_, index=X.columns)
            
        importances.nlargest(10).sort_values().plot(kind='barh', ax=axes[2], color='teal')
        axes[2].set_title('Top 10 des facteurs de survie')
    except:
        axes[2].text(0.5, 0.5, "Importance non disponible\npour ce modèle", ha='center')

    plt.tight_layout()
    plt.show()

def run_evaluation(data_path, model_path):
    print("Initialisation de l'évaluation...")
    
    # 1. Chargement des données
    if not os.path.exists(data_path):
        print(f" Erreur : Fichier de données introuvable : {data_path}")
        return

    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        print(f" Erreur lors du nettoyage : {e}")
        return

    # 2. Séparation Features/Cible
    # Utilisation d'une détection plus robuste du nom de la cible
    possible_targets = ['survie', 'survival', 'class', 'target']
    target_col = next((c for c in df.columns if c.lower() in possible_targets), df.columns[-1])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Chargement du modèle
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le fichier modèle '{model_path}' n'existe pas.")
        return

    try:
        model = joblib.load(model_path)
        print(f"✅ Modèle '{model_path}' chargé.")
    except Exception as e:
        print(f"❌ Erreur technique lors du chargement : {e}")
        return

    # 4. Lancement de l'évaluation médicale
    evaluate_medical_model(model, X, y, model_name=os.path.basename(model_path))

if __name__ == "__main__":
    # Configuration des chemins
    DATA_FILE = 'data/bone-marrow.arff' 
    MODEL_FILE = 'modele_gb.pkl' 
    
    run_evaluation(DATA_FILE, MODEL_FILE)