import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.data_processing import load_and_clean_data

def run_evaluation(data_path, model_path):
    print("--- DÉMARRAGE DE L'ÉVALUATION ---")

    # 1. Chargement des données
    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"Colonnes utilisées : {list(X.columns[:3])} ...")
    print(f"Variable cible : {y.name}")

    # 2. Chargement du modèle
    try:
        model = joblib.load(model_path)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur : Impossible de trouver {model_path}")
        return

    # 3. Prédictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # 4. Rapport de classification
    print("\n--- RÉSULTATS MÉDICAUX ---")
    print(classification_report(y, y_pred))

    # 5. Matrice de Confusion + Courbe ROC
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Performance ROC')
    plt.legend()

    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.show()

    # 6. Analyse SHAP
    print("\n--- ANALYSE SHAP ---")
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer.shap_values(X)

    # Graphique importance globale
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Importance globale des features (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_importance.png")
    plt.show()
    print("Graphique sauvegardé : shap_importance.png")

    # Graphique détaillé
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Impact des features sur la prédiction (SHAP)")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.show()
    print("Graphique sauvegardé : shap_summary.png")

    # Top 10 features
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance SHAP': np.abs(sv).mean(axis=0)
    }).sort_values('Importance SHAP', ascending=False)

    print("\n--- TOP 10 FEATURES LES PLUS IMPORTANTES ---")
    print(importance.head(10).to_string(index=False))

    return importance


if __name__ == "__main__":
    DATA_FILE = 'data/bone-marrow.arff'
    MODEL_FILE = 'modele_rf.pkl'
    run_evaluation(DATA_FILE, MODEL_FILE)
    