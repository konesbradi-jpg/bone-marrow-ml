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