import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

# Import de ta fonction de nettoyage
from data_processing import load_and_clean_data

def run_evaluation():
    print("\n" + "="*50)
    print("--- DÉMARRAGE DE L'ÉVALUATION ---")
    print("="*50)

    # 1. Chargement des données de test (sauvegardées par train_model.py)
    try:
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        print("[OK] Données de test chargées.")
    except FileNotFoundError:
        print("[ERREUR] Fichiers de test introuvables. Lancez d'abord train_model.py")
        return

    # 2. Sélection et chargement du modèle (On prend Random Forest par défaut)
    model_path = 'models/random_forest.pkl' # Tu peux changer par xgboost.pkl ou svm.pkl
    if not os.path.exists(model_path):
        print(f"[ERREUR] Modèle {model_path} introuvable.")
        return
    
    model = joblib.load(model_path)
    print(f"[OK] Modèle {model_path} chargé.")

    # 3. Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4. Rapport de classification (Precision, Recall, F1)
    print("\n--- RÉSULTATS DES MÉTRIQUES (Exigence Projet) ---")
    report = classification_report(y_test, y_pred)
    print(report)
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score : {auc_score:.4f}")

    # 5. Sauvegarde des graphiques de performance
    plt.figure(figsize=(12, 5))

    # Matrice de Confusion
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')

    # Courbe ROC
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('Courbe ROC')
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    print("\n[OK] Graphiques de performance sauvegardés : evaluation_metrics.png")

# 6. Analyse SHAP (Exigence Projet : ML EXPLICABLE)
    print("\n--- ANALYSE SHAP (Interprétabilité) ---")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Gestion de la structure des données SHAP pour Random Forest
    if isinstance(shap_values, list):
        # Pour Random Forest sklearn, on prend la classe 1 (index 1)
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Si c'est un tableau 3D, on prend les valeurs de la classe positive
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Calcul de l'importance moyenne (doit être un vecteur 1D)
    # On s'assure que le résultat est bien un tableau numpy simple
    global_importance = np.abs(sv).mean(axis=0)

    # Graphique Importance Globale
    plt.figure()
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    plt.title("Importance globale des caractéristiques (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_importance.png")
    
    # Création du tableau Top 10
    importance_df = pd.DataFrame({
        'Caractéristique': X_test.columns,
        'Importance SHAP': global_importance
    }).sort_values(by='Importance SHAP', ascending=False)

    print("\n--- TOP 10 DES CARACTÉRISTIQUES LES PLUS INFLUENTES ---")
    print(importance_df.head(10).to_string(index=False))
    
    # Création de l'explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Cas particulier pour RandomForest de sklearn (souvent renvoie une liste pour chaque classe)
    if isinstance(shap_values, list):
        # On prend les valeurs pour la classe 1 (succès de la greffe)
        sv = shap_values[1]
    else:
        sv = shap_values

    # Graphique Importance Globale
    plt.figure()
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    plt.title("Importance globale des caractéristiques (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_importance.png")
    
    # Calcul manuel du Top 10 pour l'affichage console
    importance_df = pd.DataFrame({
        'Caractéristique': X_test.columns,
        'Importance SHAP': np.abs(sv).mean(axis=0)
    }).sort_values(by='Importance SHAP', ascending=False)

    print("\n--- TOP 10 DES CARACTÉRISTIQUES LES PLUS INFLUENTES ---")
    print(importance_df.head(10).to_string(index=False))
    print("\n[OK] Analyse SHAP terminée. Graphique sauvegardé : shap_importance.png")

if __name__ == "__main__":
    run_evaluation()