import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Import de tes fonctions depuis data_processing.py
from data_processing import load_and_clean_data, optimize_memory, handle_missing_values

def run_training():
    # 1. Préparation des dossiers
    if not os.path.exists('models'): os.makedirs('models')
    
    print("\n" + "="*50)
    print("--- [1/5] CHARGEMENT ET NETTOYAGE ---")
    file_path = 'data/bone-marrow.arff'
    
    if not os.path.exists(file_path):
        print(f"ERREUR : Le fichier {file_path} est introuvable.")
        return

    # Utilisation de tes fonctions
    df = load_and_clean_data(file_path)
    df = handle_missing_values(df)
    
    print("--- [2/5] OPTIMISATION MÉMOIRE ---")
    df = optimize_memory(df)

    # 2. Préparation Features / Target
    target = 'survival_status'
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))

    X = df.drop(target, axis=1)
    y = df[target]

    # Encodage des variables catégorielles (Dummies)
    X = pd.get_dummies(X, drop_first=True)

    # 3. Division du dataset (Stratify pour garder l'équilibre des classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("--- [3/5] ÉQUILIBRAGE DES CLASSES (SMOTE) ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Sauvegarde des données de test et des noms de colonnes
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    # 4. Entraînement des 4 modèles
    print("--- [4/5] ENTRAÎNEMENT DES 4 MODÈLES ---")
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "svm": SVC(probability=True, random_state=42),
        "lightgbm": LGBMClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"Entraînement de : {name}...")
        model.fit(X_train_res, y_train_res)
        # Sauvegarde individuelle dans le dossier models/
        joblib.dump(model, f'models/{name}.pkl')

    print("--- [5/5] TERMINÉ ! ---")
    print("Tous les modèles sont sauvegardés dans le dossier 'models/'.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_training()