<<<<<<< HEAD
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
=======
"""
Évaluation des modèles — Greffe de moelle osseuse pédiatrique
=============================================================
Adapté au pipeline data_processing.py :
  - Pipelines sklearn (StandardScaler intégré) → X_test non scalé en entrée
  - Extraction des feature importances depuis pipeline.named_steps
  - Fallback decision_function pour SVM sans probability=True
  - Toutes les visualisations : confusion, ROC, importances, heatmap métriques
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── Constantes ───────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
OUTPUT_DIR  = "outputs"
MODEL_NAMES = ["randomforest", "xgboost", "svm"]

# Palette cohérente pour toutes les figures
PALETTE = {
    "randomforest": "#2196F3",
    "xgboost":      "#FF9800",
    "svm":          "#4CAF50",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES DE TEST
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(
    x_path: str = "data/X_test.csv",
    y_path: str = "data/y_test.csv",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge X_test et y_test produits par data_processing.full_pipeline().
    y_test peut être un DataFrame 1-colonne ou une Series — on normalise.
    """
    X_test = pd.read_csv(x_path)
    y_raw  = pd.read_csv(y_path)

    # pd.read_csv retourne toujours un DataFrame — on extrait la Series
    y_test = y_raw.iloc[:, 0].astype(int)

    print(f"✅ Test set chargé : {X_test.shape[0]} exemples, {X_test.shape[1]} features")
    print(f"   Distribution cible : {y_test.value_counts().to_dict()}")
    return X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTIONS — COMPATIBLE PIPELINE ET MODÈLE BRUT
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (y_pred, y_score) pour n'importe quel modèle ou Pipeline sklearn.

    Stratégie :
      1. predict_proba  → RF, XGBoost, SVM avec probability=True
      2. decision_function → SVM sans probability (normalisé 0–1 pour ROC-AUC)
      3. Fallback binaire  → dernier recours

    Le modèle peut être un Pipeline(scaler, estimator) ou un estimateur brut.
    Dans les deux cas, model.predict() et model.predict_proba() fonctionnent
    identiquement — le Pipeline applique le scaler automatiquement.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):
        # SVM sans probability=True
        raw = model.decision_function(X_test)
        # Normalisation min-max → [0, 1] pour que le ROC-AUC soit interprétable
        y_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    else:
        y_score = y_pred.astype(float)

    return y_pred, y_score


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DES FEATURE IMPORTANCES DEPUIS UN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importances(
    model, feature_names: list[str]
) -> pd.Series | None:
    """
    Extrait les importances de features que le modèle soit un Pipeline
    sklearn ou un estimateur brut.

    Pour un Pipeline, l'estimateur final est dans model.steps[-1][1].
    RF et XGBoost exposent feature_importances_.
    SVM ne l'expose pas → retourne None.
    """
    # Cas Pipeline sklearn
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]   # dernier étage du Pipeline

    if hasattr(estimator, "feature_importances_"):
        return (
            pd.Series(estimator.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_models(
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Charge chaque modèle depuis models/<n>_model.pkl et calcule :
      - Les métriques de classification
      - Les importances de features (RF et XGBoost)
      - Les données pour les courbes ROC

    Retourne
    --------
    df_results  : DataFrame des métriques (indexé par nom de modèle)
    importances : dict {name: pd.Series}
    roc_data    : dict {name: (fpr, tpr, auc_score)}
    """
    results     = []
    importances = {}
    roc_data    = {}

    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")

        if not os.path.exists(path):
            print(f"⚠️  Modèle introuvable : {path} — ignoré.")
            continue

        print(f"\n→ Évaluation : {name.upper()}")
        model = joblib.load(path)
        y_pred, y_score = get_predictions(model, X_test)

        metrics = {
            "Modèle"    : name.upper(),
            "Accuracy"  : round(accuracy_score(y_test, y_pred), 4),
            "Precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score"  : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC"   : round(roc_auc_score(y_test, y_score), 4),
        }
        results.append(metrics)
        print(f"   {metrics}")

        # Feature importances (RF, XGBoost — pas SVM)
        imp = extract_feature_importances(model, X_test.columns.tolist())
        if imp is not None:
            importances[name] = imp

        # Données ROC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    df_results = pd.DataFrame(results).set_index("Modèle")
    return df_results, importances, roc_data


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | None = None,
) -> None:
    """Matrices de confusion côte à côte pour tous les modèles disponibles."""
    models_loaded = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            models_loaded[name] = joblib.load(path)

    n = len(models_loaded)
    if n == 0:
        print("⚠️  Aucun modèle trouvé pour les matrices de confusion.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_loaded.items()):
        y_pred, _ = get_predictions(model, X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["Survie (0)", "Décès (1)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.upper(), fontsize=13, fontweight="bold")

    fig.suptitle("Matrices de Confusion", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = save_path or os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Matrices de confusion → {out}")
    plt.close()


def plot_roc_curves(
    roc_data: dict,
    save_path: str | None = None,
) -> None:
    """Courbes ROC superposées pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Aléatoire (AUC = 0.50)")

    for name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(
            fpr, tpr,
            color=PALETTE.get(name, "gray"),
            lw=2.5,
            label=f"{name.upper()} (AUC = {auc_score:.3f})",
        )

    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
    ax.set_title("Courbes ROC — Comparaison des modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    out = save_path or os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Courbes ROC → {out}")
    plt.close()


def plot_feature_importances(
    importances: dict,
    top_n: int = 15,
    save_path: str | None = None,
) -> None:
    """
    Barplots horizontaux des top_n features les plus importantes.
    Un subplot par modèle (RF et XGBoost uniquement — SVM non supporté).
    """
    if not importances:
        print("⚠️  Aucune importance de feature disponible.")
        return

    bar_palettes = {"randomforest": "Blues_d", "xgboost": "Oranges_d"}
    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, importances.items()):
        top = series.head(top_n)
        sns.barplot(
            x=top.values, y=top.index,
            palette=bar_palettes.get(name, "viridis"),
            ax=ax,
        )
        ax.set_title(
            f"{name.upper()} — Top {top_n} Features",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_ylabel("")

    plt.tight_layout()
    out = save_path or os.path.join(OUTPUT_DIR, "feature_importances.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Feature importances → {out}")
    plt.close()


def plot_metrics_heatmap(
    df_results: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Heatmap du tableau comparatif — vue d'ensemble rapide du meilleur modèle."""
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(
        df_results.astype(float),
        annot=True, fmt=".3f",
        cmap="YlGn", linewidths=0.5,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title("Comparaison des métriques", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = save_path or os.path.join(OUTPUT_DIR, "metrics_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Heatmap métriques → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("      ÉVALUATION DES MODÈLES — BONE MARROW")
    print("=" * 55)

    # 1. Charger les données de test (produites par data_processing.py)
    X_test, y_test = load_test_data("data/X_test.csv", "data/y_test.csv")

    # 2. Évaluer tous les modèles
    df_results, importances, roc_data = evaluate_all_models(X_test, y_test)

    # 3. Afficher le tableau
    print("\n📋 Tableau Comparatif :")
    print(df_results.to_string())

    # 4. Sauvegarder le CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df_results.to_csv(csv_path)
    print(f"\n💾 Résultats → {csv_path}")

    # 5. Générer toutes les visualisations
    plot_confusion_matrices(X_test, y_test)
    plot_roc_curves(roc_data)
    plot_feature_importances(importances)
    plot_metrics_heatmap(df_results)

    print("\n✅ Évaluation complète terminée.")
>>>>>>> 66debb900212697f87d49a69e294f90032729b57
