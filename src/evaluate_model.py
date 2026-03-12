import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def run_evaluation():
    print("\n" + "="*70)
    print("--- DÉMARRAGE DE L'ÉVALUATION COMPARATIVE ---")
    print("="*70)

    # 1. Chargement des données de test
    try:
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv').values.ravel() # Assurer le format 1D
        print("[OK] Données de test chargées.")
    except FileNotFoundError:
        print("[ERREUR] Fichiers de test introuvables. Lancez train_model.py d'abord.")
        return

    # 2. Liste des modèles à tester
    model_names = ["random_forest", "xgboost", "svm", "lightgbm"]
    results = []
    trained_models = {}

    print(f"\n{'Modèle':<15} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}")
    print("-" * 70)

    for name in model_names:
        path = f'models/{name}.pkl'
        if os.path.exists(path):
            model = joblib.load(path)
            trained_models[name] = model
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Probabilités pour l'AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            # Calcul des métriques
            metrics = {
                "Modèle": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }
            results.append(metrics)

            print(f"{name:<15} | {metrics['Accuracy']:.3f} | {metrics['Precision']:.3f} | "
                  f"{metrics['Recall']:.3f} | {metrics['F1']:.3f} | {metrics['ROC-AUC']:.3f}")

    # 3. Choisir le meilleur modèle (Basé sur le ROC-AUC)
    df_results = pd.DataFrame(results)
    best_model_name = df_results.sort_values(by="ROC-AUC", ascending=False).iloc[0]["Modèle"]
    best_model = trained_models[best_model_name]
    
    print("\n" + "="*70)
    print(f"🏆 LE MEILLEUR MODÈLE EST : {best_model_name.upper()}")
    print("="*70)

    # Sauvegarder le meilleur modèle pour l'interface Streamlit
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"[OK] Meilleur modèle sauvegardé sous : models/best_model.pkl")

    # 4. Analyse SHAP pour le MEILLEUR MODÈLE
    print(f"\n--- ANALYSE SHAP POUR {best_model_name.upper()} ---")
    
    # Utilisation d'un explainer générique pour plus de stabilité
    try:
        if best_model_name == "svm":
            # SVM nécessite un explainer spécifique ou un échantillon plus petit
            explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_test, 10))
            sv = explainer.shap_values(X_test)
        else:
            # Pour les modèles à base d'arbres (RF, XGB, LGBM)
            explainer = shap.TreeExplainer(best_model)
            sv = explainer.shap_values(X_test)

        # Gestion de la structure SHAP (classe 1)
        if isinstance(sv, list):
            sv = sv[1]
        elif len(sv.shape) == 3:
            sv = sv[:, :, 1]

        # Graphique Importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_test, plot_type="bar", show=False)
        plt.title(f"Importance des variables - {best_model_name}")
        plt.tight_layout()
        plt.savefig("shap_importance.png")
        
        # Top 10 console
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': np.abs(sv).mean(axis=0)
        }).sort_values(by='Importance', ascending=False)

        print("\n--- TOP 10 FEATURES (FACTEURS MÉDICAUX CLÉS) ---")
        print(importance_df.head(10).to_string(index=False))
        print("\n[OK] Graphique SHAP sauvegardé : shap_importance.png")

    except Exception as e:
        print(f"[ATTENTION] Erreur SHAP : {e}. Cela arrive parfois avec certains modèles.")

    # 5. Sauvegarde des métriques finales
    df_results.to_csv('models/evaluation_results.csv', index=False)

if __name__ == "__main__":
    run_evaluation()