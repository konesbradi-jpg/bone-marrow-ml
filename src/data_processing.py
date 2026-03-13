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