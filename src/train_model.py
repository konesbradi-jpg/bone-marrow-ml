<<<<<<< HEAD
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Liste des colonnes de "Fuite de données" à supprimer impérativement
# Ces colonnes donnent la réponse au modèle avant qu'il ne réfléchisse
COLS_TO_DROP = ['survival_time', 'time_to_aGvHD_III_IV', 'PLTrecovery', 'ANCrecovery']

def train_and_save_models():
    if not os.path.exists('models'):
        os.makedirs('models')

    print("[1/5] Chargement des données...")
    try:
        # Remplacez par le nom exact de votre fichier
        df = pd.read_csv('data/bone-marrow.csv') 
    except FileNotFoundError:
        print("Erreur : Fichier CSV introuvable dans 'data/'")
        return

    # --- ÉTAPE CRUCIALE : NETTOYAGE ---
    # Remplacer 'target' par le nom réel de votre colonne cible (ex: 'survival_status')
    TARGET_COL = 'survival_status' 
    
    if TARGET_COL not in df.columns:
        # Si vous n'avez pas renommé votre colonne cible, on essaie de la deviner
        # Souvent c'est la dernière colonne ou une colonne binaire
        print(f"Attention: {TARGET_COL} non trouvé. Vérifiez le nom de la colonne cible.")
        return

    # Suppression des fuites de données
    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Gestion des valeurs manquantes (médiane pour le numérique)
    df = df.replace('?', np.nan) # Si le dataset contient des '?'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # --- ÉTAPE CRUCIALE : ENCODAGE ---
    # On transforme les catégories en chiffres
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int)

    # On utilise get_dummies mais on sauvegarde l'ordre des colonnes !
    X = pd.get_dummies(X)
    
    # On sauvegarde la liste EXACTE des colonnes pour Streamlit
    model_features = X.columns.tolist()
    joblib.dump(model_features, 'models/features.pkl')

    # 4. Division et SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[2/5] Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Entraînement
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "xgboost": XGBClassifier(random_state=42),
        "best_model": RandomForestClassifier(n_estimators=200, random_state=42) # Modèle par défaut pour l'interface
    }

    print("[3/5] Entraînement et validation...")
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        print(f"\n--- Rapport pour {name} ---")
        print(classification_report(y_test, y_pred))
        
        joblib.dump(model, f'models/{name}.pkl')
    
    print("\n[4/5] Succès ! Modèles et liste des features sauvegardés.")

if __name__ == "__main__":
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
