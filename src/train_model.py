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
    train_medical_model()