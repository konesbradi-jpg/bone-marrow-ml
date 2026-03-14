<<<<<<< HEAD
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
TARGET = 'survival_status'  # 1 = Mort, 0 = Survie
COLS_TO_DROP = ['survival_time', 'time_to_aGvHD_III_IV', 'PLTrecovery', 'ANCrecovery', 'extcGvHD', 'aGvHDIIIIV']

def build_robust_pipeline(X_train, y_train):
    """
    Pipeline configuré pour prioriser la détection de la classe 1 (Mort).
    """
    # Calcul du ratio de déséquilibre
    counts = np.bincount(y_train)
    # Ratio = Nombre de survies (0) / Nombre de morts (1)
    # Si ratio = 5, une erreur sur une 'Mort' coûtera 5x plus cher qu'une erreur sur une 'Survie'
    weight_ratio = counts[0] / counts[1]
    
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler()),
        ('resampler', SMOTETomek(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            # On donne beaucoup plus de poids à la classe 1 (Mort)
            class_weight={1: weight_ratio * 2, 0: 1}, 
            random_state=42
        ))
    ])
    return pipeline

def train_medical_model():
    if not os.path.exists('models'): os.makedirs('models')

    # 1. Chargement et Nettoyage
    print("[1/4] Chargement et suppression du leakage...")
    df = pd.read_csv('data/bone-marrow.csv').replace('?', np.nan)
    
    # Supprimer les colonnes de fuite (le futur)
    drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drops)
    
    # 2. Préparation des données
    X = df.drop(TARGET, axis=1)
    y = df[TARGET].astype(int)
    
    # Encodage catégoriel
    X = pd.get_dummies(X)
    
    # Division Stratifiée (très important pour les petits datasets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sauvegarde des métadonnées
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    # 3. Entraînement du Pipeline
    print(f"[2/4] Entraînement sur {len(X_train)} patients (Cible 1 = Mort)...")
    model_pipeline = build_robust_pipeline(X_train, y_train)
    model_pipeline.fit(X_train, y_train)

    # 4. Évaluation spécialisée
    y_pred = model_pipeline.predict(X_test)
    print("\n=== RAPPORT DE PERFORMANCE MÉDICALE ===")
    print(classification_report(y_test, y_pred, target_names=['Survie (0)', 'Mort (1)']))
    
    print("\nMATRICE DE CONFUSION :")
    print(confusion_matrix(y_test, y_pred))

    # Sauvegarde
    joblib.dump(model_pipeline, 'models/best_model.pkl')
    print("\n[4/4] Pipeline sauvegardé avec succès.")

if __name__ == "__main__":
<<<<<<< HEAD
    train_medical_model()
=======
    train_and_save_models()
=======
"""
Entraînement des modèles — Greffe de moelle osseuse pédiatrique
===============================================================
Colonnes exclues :
  - 8 colonnes POST-GREFFE (data leakage) : observables seulement après la greffe
  - 5 colonnes REDONDANTES : versions binarisées de variables continues déjà présentes
    (garder les deux crée de la multicolinéarité sans apporter d'info supplémentaire)
"""

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_processing import full_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# COLONNES À EXCLURE
# ─────────────────────────────────────────────────────────────────────────────

# Observables APRÈS la greffe → le modèle ne peut pas les connaître en pratique
LEAKAGE_COLS = [
    "IIIV",                  # GvHD aiguë stade II/III/IV
    "Relapse",               # Rechute de la maladie
    "aGvHDIIIIV",            # GvHD aiguë stade III/IV
    "extcGvHD",              # GvHD chronique extensive
    "ANCrecovery",           # Temps de récupération des neutrophiles
    "PLTrecovery",           # Temps de récupération des plaquettes
    "time_to_aGvHD_III_IV",  # Délai avant GvHD III/IV
    "survival_time",         # Durée de survie (leakage évident)
]

# Versions binarisées de variables continues déjà présentes :
# garder les deux introduit de la multicolinéarité sans info supplémentaire
# Ex : Donorage (continu) + Donorage35 (0/1) → on garde Donorage
REDUNDANT_COLS = [
    "Donorage35",      # binarisé de Donorage
    "Recipientage10",  # binarisé de Recipientage
    "Recipientageint", # intervalles de Recipientage
    "HLAmismatch",     # binarisé de HLAmatch
    "Diseasegroup",    # binarisé de Disease
]

COLS_TO_DROP = LEAKAGE_COLS + REDUNDANT_COLS

MODELS_DIR = "models"
DATA_DIR   = "data"
ARFF_PATH  = os.path.join(DATA_DIR, "bone-marrow.arff")


# ─────────────────────────────────────────────────────────────────────────────
# SUPPRESSION DES COLONNES INUTILES / DANGEREUSES
# ─────────────────────────────────────────────────────────────────────────────

def drop_invalid_cols(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Supprime les colonnes post-greffe et redondantes de X_train et X_test.
    Gère aussi les variantes OHE (ex: IIIV_1, extcGvHD_0).
    Vérifie que train et test ont exactement les mêmes colonnes supprimées.
    """
    def _find_cols(df: pd.DataFrame) -> list[str]:
        to_drop = []
        for col in COLS_TO_DROP:
            matched = [c for c in df.columns if c == col or c.startswith(f"{col}_")]
            to_drop.extend(matched)
        return to_drop

    drop_train = _find_cols(X_train)
    drop_test  = _find_cols(X_test)

    assert set(drop_train) == set(drop_test), (
        f"Colonnes asymétriques entre train et test !\n"
        f"  Train: {sorted(drop_train)}\n  Test:  {sorted(drop_test)}"
    )

    leakage_found   = [c for c in drop_train if any(c == l or c.startswith(f"{l}_") for l in LEAKAGE_COLS)]
    redundant_found = [c for c in drop_train if any(c == r or c.startswith(f"{r}_") for r in REDUNDANT_COLS)]

    print(f"🚫 Post-greffe supprimées  ({len(leakage_found)}) : {leakage_found}")
    print(f"🔁 Redondantes supprimées  ({len(redundant_found)}) : {redundant_found}")
    print(f"✅ Features finales : {X_train.shape[1] - len(drop_train)} colonnes")

    return X_train.drop(columns=drop_train), X_test.drop(columns=drop_test)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES SKLEARN
# ─────────────────────────────────────────────────────────────────────────────

def build_pipelines() -> dict[str, Pipeline]:
    """
    Pipeline sklearn pour chaque modèle : StandardScaler + estimateur.

    Pourquoi des Pipelines ?
    - Le scaler est OBLIGATOIRE pour SVM (distances euclidiennes)
    - Encapsulé dans le Pipeline, model.predict(X_test) applique le scaler
      automatiquement → X_test.csv reste dans l'espace original, pas de
      risque d'oublier de scaler dans evaluate_model.py
    - RF et XGBoost sont invariants au scaling mais le Pipeline les rend
      cohérents et facilite la sérialisation joblib
    """
    return {
        "randomforest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, random_state=42)),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT ET SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # 1. Preprocessing complet
    print("=" * 55)
    print("  PRÉPROCESSING")
    print("=" * 55)
    X_train, X_test, y_train, y_test = full_pipeline(ARFF_PATH, save_dir=None)

    # 2. Suppression leakage + redondances
    print("\n" + "=" * 55)
    print("  NETTOYAGE DES FEATURES")
    print("=" * 55)
    X_train, X_test = drop_invalid_cols(X_train, X_test)

    # 3. Sauvegarde X_test / y_test propres (après suppression)
    X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)
    joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, "features_list.pkl"))
    print(f"💾 X_test, y_test, features_list → '{DATA_DIR}/'")

    # 4. Entraînement
    print("\n" + "=" * 55)
    print("  ENTRAÎNEMENT")
    print("=" * 55)
    for name, pipeline in build_pipelines().items():
        print(f"\n→ {name.upper()}...")
        pipeline.fit(X_train, y_train)
        out = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        joblib.dump(pipeline, out)
        print(f"   ✅ Sauvegardé → {out}")

    print("\n" + "=" * 55)
    print("  TERMINÉ — 3 modèles prêts dans 'models/'")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_save()
>>>>>>> 66debb900212697f87d49a69e294f90032729b57
>>>>>>> aa03af1db19de3a56e8ef6eb5c0c5ea4ae097ef3
