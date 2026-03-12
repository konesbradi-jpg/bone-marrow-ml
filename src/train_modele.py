import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Importation de tes fonctions de nettoyage
from data_processing import load_and_preprocess, optimize_memory

def train_and_save_models():
    # 1. Création du dossier pour les modèles s'il n'existe pas
    if not os.path.exists('models'):
        os.makedirs('models')

    # 2. Chargement des données
    # REMPLACE 'data/dataset.csv' par le nom exact de ton fichier
    print("[1/5] Chargement et optimisation des données...")
    try:
        df = pd.read_csv('data/bone-marrow.csv') # Vérifie le nom du fichier !
    except FileNotFoundError:
        print("Erreur : Place le fichier CSV dans le dossier 'data/'")
        return

    # Optimisation mémoire (exigence PDF)
    df, start_mem, end_mem = optimize_memory(df)
    print(f"Mémoire optimisée : {start_mem:.2f}MB -> {end_mem:.2f}MB")

    # 3. Prétraitement simple (à adapter selon ton dataset)
    # On suppose que la colonne cible est 'target'
    # Supprime les colonnes inutiles ou gère les valeurs manquantes
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop('target', axis=1) # Remplace 'target' par le nom de ta colonne
    y = df['target']
    
    # Encodage des variables catégorielles (ex: sexe, type de maladie)
    X = pd.get_dummies(X, drop_first=True)

    # 4. Division et Gestion du déséquilibre avec SMOTE (exigence PDF)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[2/5] Application de SMOTE pour équilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Sauvegarde des données de test pour evaluate_model.py
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    # Sauvegarde des colonnes pour l'interface Streamlit
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    # 5. Entraînement des modèles
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "lightgbm": LGBMClassifier(random_state=42)
    }

    print("[3/5] Entraînement des modèles (cela peut prendre un moment)...")
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train_res, y_train_res)
        # Sauvegarde de chaque modèle en format .pkl
        joblib.dump(model, f'models/{name}.pkl')
    
    print("[4/5] Modèles sauvegardés avec succès dans le dossier 'models/'")

if __name__ == "__main__":
    train_and_save_models()
