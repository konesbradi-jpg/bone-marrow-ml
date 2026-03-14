<<<<<<< HEAD
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, balanced_accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. CONFIGURATION
TARGET = 'survival_status'
# On retire tout ce qui arrive APRÈS la greffe (Data Leakage)
COLS_TO_DROP = ['survival_time', 'time_to_aGvHD_III_IV', 'PLTrecovery', 'ANCrecovery', 'extcGvHD', 'aGvHDIIIIV']

def build_robust_pipeline(X_train, y_train):
    """
    Construit un pipeline qui gère les trous, l'échelle, 
    le déséquilibre et l'entraînement de manière isolée.
    """
    
    # Calcul automatique du poids des classes pour compenser le déséquilibre
    # Si on a 80% survie / 20% mort, le poids de la mort sera de 4.
    counts = np.bincount(y_train)
    weight_ratio = counts[1] / counts[0]
    
    pipeline = Pipeline([
        # A. Remplissage des trous (KNN cherche les voisins proches pour deviner)
        ('imputer', KNNImputer(n_neighbors=5)),
        
        # B. Mise à l'échelle robuste (ne craint pas les valeurs extrêmes de poids ou d'âge)
        ('scaler', RobustScaler()),
        
<<<<<<< HEAD
        # C. Rééquilibrage hybride (SMOTE crée + Tomek Links nettoie)
        ('resampler', SMOTETomek(random_state=42)),
        
        # D. Modèle avec poids de classe intégré
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            max_depth=7,            # Profondeur limitée pour petit dataset (évite l'overfitting)
            class_weight={0: weight_ratio * 1.5, 1: 1}, # On sur-pénalise l'erreur sur le décès
            random_state=42
        ))
    ])
    
    return pipeline

def train_medical_model():
    if not os.path.exists('models'): os.makedirs('models')

    # Chargement
    df = pd.read_csv('data/bone-marrow.csv').replace('?', np.nan)
    
    # Nettoyage
    drops = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drops)
    
    # Encodage (One-Hot)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET].astype(int)
    X = pd.get_dummies(X)
    
    # Division Stratifiée (garde la même proportion de morts dans train et test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sauvegarde des colonnes pour Streamlit
    joblib.dump(X.columns.tolist(), 'models/features.pkl')
    # Sauvegarde des valeurs par défaut pour l'imputation manuelle au cas où
    joblib.dump(X_train.median(), 'models/impute_values.pkl')

    print(f"Entraînement sur {len(X_train)} patients...")
    
    # Création et entraînement
    model_pipeline = build_robust_pipeline(X_train, y_train)
    model_pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred = model_pipeline.predict(X_test)
    
    print("\n=== RAPPORT DE PERFORMANCE MÉDICALE ===")
    print(classification_report(y_test, y_pred))
    print(f"Score équilibré : {balanced_accuracy_score(y_test, y_pred):.2%}")

    # Sauvegarde du pipeline complet (incluant l'imputer et le scaler)
    joblib.dump(model_pipeline, 'models/best_model.pkl')
    print("\nPipeline sauvegardé dans models/best_model.pkl")

if __name__ == "__main__":
    train_medical_model()
=======
    return df
=======
"""
Pipeline de prétraitement — Greffe de moelle osseuse pédiatrique
================================================================
Améliorations par rapport à la version originale :
  - Chargement natif du fichier .arff (scipy)
  - Séparation claire des colonnes numériques / catégorielles / binaires
  - Imputation ciblée : KNN pour le numérique, mode pour le catégoriel
    (le KNN sur des données encodées en entiers est sémantiquement non-fiable)
  - optimize_memory APRÈS l'imputation pour ne pas dégrader la précision du KNN
  - Encodage OHE propre avant le split (pas après)
  - SMOTE appliqué uniquement sur le train set (pas de fuite)
  - Logging clair à chaque étape
"""

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_arff(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier .arff et décode les bytes en str.
    Retourne un DataFrame pandas propre.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Les colonnes catégorielles sont encodées en bytes par scipy → on décode
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")

    log.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    log.info(f"Valeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Impute les valeurs manquantes selon le type de colonne :
      - Numérique  → KNNImputer (sur données scalées, puis inverse)
      - Catégoriel → SimpleImputer(strategy='most_frequent')

    Pourquoi ne PAS utiliser KNN sur les catégorielles ?
    Le KNN calcule des distances euclidiennes. Encoder 'A'→0, 'B'→1, 'AB'→2
    implique que AB est "deux fois plus loin de A que B", ce qui n'a pas de
    sens médical. Le mode est plus robuste pour ce type de variable.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # "str" explicite requis pour pandas 4+ (include="object" déprécié pour str)
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()

    missing_num = df[num_cols].isnull().sum()
    missing_cat = df[cat_cols].isnull().sum()
    log.info(f"Colonnes numériques avec NaN : {missing_num[missing_num > 0].to_dict()}")
    log.info(f"Colonnes catégorielles avec NaN : {missing_cat[missing_cat > 0].to_dict()}")

    # --- Numérique : KNN sur données normalisées ---
    if df[num_cols].isnull().any().any():
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[num_cols])
        imputed_scaled = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
        df[num_cols] = scaler.inverse_transform(imputed_scaled)
        log.info(f"KNN imputation appliquée sur : {num_cols}")

    # --- Catégoriel : mode ---
    if df[cat_cols].isnull().any().any():
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        log.info(f"Mode imputation appliquée sur les colonnes catégorielles.")

    assert df.isnull().sum().sum() == 0, "Il reste des NaN après imputation !"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTIMISATION MÉMOIRE  (après imputation pour préserver la précision KNN)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire APRÈS l'imputation.
    float64 → float32, int64 → int32
    Note : on ne descend pas à float16 car les colonnes médicales continues
    (CD34kgx10d6, survival_time…) nécessitent la précision float32 minimum.
    """
    before = df.memory_usage(deep=True).sum() / 1024
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    after = df.memory_usage(deep=True).sum() / 1024
    log.info(f"Mémoire : {before:.1f} KB → {after:.1f} KB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRÉPARATION DE LA CIBLE
# ─────────────────────────────────────────────────────────────────────────────

TARGET_COL = "survival_status"

def prepare_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    - Vérifie que la colonne cible existe
    - Renomme en 'target' pour uniformiser le reste du pipeline
    - Convertit en entier (0 = survie, 1 = décès)
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    df = df.rename(columns={target_col: "target"})
    df["target"] = df["target"].astype(int)
    log.info(f"Distribution de la cible :\n{df['target'].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENCODAGE + SPLIT + SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def encode_split_resample(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Étapes dans le bon ordre :
      1. One-Hot Encoding (drop_first=True pour éviter la multicolinéarité)
      2. Train / Test split stratifié
      3. SMOTE sur le TRAIN uniquement
         → évite la data leakage : le test doit rester 100% réel

    Retourne : X_train_res, X_test, y_train_res, y_test
    """
    # 1. OHE — on exclut la cible
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if "target" in cat_cols:
        cat_cols.remove("target")

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # S'assurer que la cible n'a pas été transformée par get_dummies
    assert "target" in df_encoded.columns, "La colonne 'target' a disparu après OHE !"

    X = df_encoded.drop("target", axis=1)
    y = df_encoded["target"]

    # 2. Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(f"Split : {len(X_train)} train / {len(X_test)} test")
    log.info(f"Distribution train avant SMOTE : {y_train.value_counts().to_dict()}")

    # 3. SMOTE sur le train uniquement
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(f"Distribution train après SMOTE  : {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(
    filepath: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline complet :
      load → impute → optimize_memory → prepare_target → encode/split/SMOTE
    """
    df = load_arff(filepath)
    df = impute(df)
    df = optimize_memory(df)   # ← APRÈS imputation (précision KNN préservée)
    df = prepare_target(df)
    return encode_split_resample(df)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = full_pipeline(
        "/home/claude/data/bone-marrow.arff"
    )
    print(f"\n✅ Pipeline terminé")
    print(f"   X_train : {X_train.shape}, y_train : {len(y_train)}")
    print(f"   X_test  : {X_test.shape},  y_test  : {len(y_test)}")
>>>>>>> 66debb900212697f87d49a69e294f90032729b57
>>>>>>> aa03af1db19de3a56e8ef6eb5c0c5ea4ae097ef3
