import pandas as pd
import numpy as np
import joblib
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.data_processing import load_and_clean_data, handle_missing_values, optimize_memory

# --- 1. CHARGEMENT ET PREPROCESSING ---
print("Chargement des données...")
df = load_and_clean_data('data/bone-marrow.arff')
df = handle_missing_values(df)
df = optimize_memory(df)

# Séparation features / cible
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encodage des colonnes texte
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

print(f"Dataset : {X.shape[0]} patients, {X.shape[1]} features")
print(f"Distribution cible : {y.value_counts().to_dict()}")

# --- 2. SPLIT TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 3. SMOTE ---
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\nAprès SMOTE : {pd.Series(y_train_res).value_counts().to_dict()}")

# --- 4. DÉFINITION DES MODÈLES ---
modeles = {
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
}

# --- 5. ENTRAÎNEMENT ET ÉVALUATION ---
print("\n--- ÉVALUATION DES MODÈLES ---")
resultats = {}

for nom, pipeline in modeles.items():
    print(f"\nEntraînement : {nom}...")
    pipeline.fit(X_train_res, y_train_res)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {auc:.3f}")
    
    resultats[nom] = auc
    joblib.dump(pipeline, f"modele_{nom.lower().replace(' ', '_')}.pkl")
    print(f"Modèle sauvegardé : modele_{nom.lower().replace(' ', '_')}.pkl")

# --- 6. MEILLEUR MODÈLE ---
meilleur = max(resultats, key=resultats.get)
print(f"\n--- MEILLEUR MODÈLE : {meilleur} (AUC={resultats[meilleur]:.3f}) ---")

import shutil
shutil.copy(
    f"modele_{meilleur.lower().replace(' ', '_')}.pkl",
    "modele_final.pkl"
)
print("Meilleur modèle sauvegardé comme modele_final.pkl")
