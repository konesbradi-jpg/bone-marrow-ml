import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "bone-marrow.csv")

TARGET = 'survival_status'

# Colonnes de fuite (données connues après la greffe) et redondantes
LEAKAGE_COLS = ["survival_time", "time_to_aGvHD_III_IV", "PLTrecovery", "ANCrecovery", "extcGvHD", "aGvHDIIIIV", "IIIV", "Relapse"]
REDUNDANT_COLS = ["Donorage35", "Recipientage10", "Recipientageint", "HLAmismatch", "Diseasegroup"]
COLS_TO_DROP = LEAKAGE_COLS + REDUNDANT_COLS

def clean_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Erreur : Le fichier de données est introuvable à {path}")
    
    # Chargement
    df = pd.read_csv(path).replace('?', np.nan)
    
    # Nettoyage des chaînes de caractères (format b'AML' -> AML)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r"^b['\"]", "", regex=True).str.replace(r"['\"]$", "", regex=True)
        df[col] = df[col].replace(['nan', 'None'], np.nan)
    
    # Gestion de la cible (Suppression des lignes sans cible)
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').astype(int)
    
    # Suppression des colonnes inutiles
    cols_to_remove = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_remove)
    
    return df

def train_medical_models():
    # Création des dossiers si nécessaire
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Chargement et nettoyage
    print("--- [1/4] Nettoyage des données ---")
    df = clean_data(DATA_PATH)
    
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 2. Identification des types de colonnes
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Sauvegarde des infos de colonnes pour app.py
    joblib.dump({
        'features': X.columns.tolist(),
        'numeric': numeric_features,
        'categorical': categorical_features
    }, os.path.join(MODELS_DIR, 'features_info.pkl'))

    # 3. Création du Pipeline de Prétraitement
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    # Pipeline complet (Prétraitement -> Équilibrage -> Classifieur)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('resampler', SMOTETomek(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42))
    ])

    # 4. Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- ÉTAPE CRUCIALE : SAUVEGARDE DES DONNÉES DE TEST ---
    # Pour que l'évaluation fonctionne plus tard
    X_test.to_csv(os.path.join(DATA_DIR, "X_test_cleaned.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "y_test_cleaned.csv"), index=False)
    print(f"✅ Données de test sauvegardées dans {DATA_DIR}")

    # 5. Entraînement
    print("--- [2/4] Entraînement du modèle (Random Forest) ---")
    full_pipeline.fit(X_train, y_train)
    
    # 6. Évaluation rapide
    print("\n✅ Rapport de performance sur le Test Set :")
    y_pred = full_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 7. Sauvegarde du modèle
    model_file = os.path.join(MODELS_DIR, 'best_model.pkl')
    # On sauvegarde aussi une copie sous le nom générique utilisé par evaluate_model.py
    joblib.dump(full_pipeline, model_file)
    joblib.dump(full_pipeline, os.path.join(MODELS_DIR, 'randomforest_model.pkl'))
    
    print(f"\n💾 Modèles sauvegardés dans : {MODELS_DIR}")
    print("--- [4/4] Processus terminé ---")

if __name__ == "__main__":
    train_medical_models()