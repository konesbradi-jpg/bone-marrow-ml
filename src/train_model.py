import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Import de tes propres fonctions depuis data_processing.py
from data_processing import load_and_clean_data, handle_missing_values, optimize_memory

def run_training():
    # 1. Préparation des dossiers
    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('data'): os.makedirs('data')

    # 2. Chargement et Traitement initial
    print("--- [1/5] Chargement et Nettoyage ---")
    # Vérifie que ton fichier est bien dans data/bone-marrow.arff
    file_path = 'data/bone-marrow.arff'
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    df = load_and_clean_data(file_path)
    df = handle_missing_values(df)
    
    # 3. Optimisation mémoire (Exigence du projet)
    print("--- [2/5] Optimisation Mémoire ---")
    df = optimize_memory(df)

    # 4. Préparation Features / Target
    # Dans ce dataset UCI, la cible est 'survival_status'
    target = 'survival_status'
    
    # Encodage de la cible (convertir '0'/'1' en nombres 0 et 1)
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))

    X = df.drop(target, axis=1)
    y = df[target]

    # Encodage des variables catégorielles (Dummies) pour que les modèles puissent lire
    X = pd.get_dummies(X, drop_first=True)

    # 5. Division du dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Gestion du déséquilibre avec SMOTE (Exigence du projet)
    print("--- [3/5] Application de SMOTE (Équilibrage des classes) ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Sauvegarde des données de test pour evaluate_model.py et des colonnes pour l'interface
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    # 7. Entraînement des modèles (On choisit les 3 recommandés)
    print("--- [4/5] Entraînement des modèles ---")
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "svm": SVC(probability=True, random_state=42)
    }

    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train_res, y_train_res)
        # Sauvegarde du modèle au format .pkl
        joblib.dump(model, f'models/{name}.pkl')

    print("--- [5/5] Terminé ! Tous les modèles sont dans le dossier 'models/' ---")

if __name__ == "__main__":
    run_training()