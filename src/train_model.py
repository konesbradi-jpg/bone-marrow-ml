import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Import de tes propres fonctions
from data_processing import load_and_clean_data, optimize_memory, handle_missing_values

def run_training():
    if not os.path.exists('models'): os.makedirs('models')
    
    print("--- [1/5] Chargement et Nettoyage ---")
    file_path = 'data/bone-marrow.arff'
    df = load_and_clean_data(file_path)
    df = handle_missing_values(df)
    
    print("--- [2/5] Optimisation Mémoire ---")
    df = optimize_memory(df)

    # Cible du dataset
    target = 'survival_status'
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target].astype(str))

    X = df.drop(target, axis=1)
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)

    # Split et SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("--- [3/5] Application de SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Sauvegarde pour l'évaluation
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    print("--- [4/5] Entraînement du modèle (Random Forest) ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # On le sauvegarde sous le nom attendu par evaluate_model.py
    joblib.dump(model, 'models/random_forest.pkl')
    # Optionnel : On peut aussi sauvegarder en tant que modele_rf.pkl pour ton interface
    joblib.dump(model, 'modele_rf.pkl')

    print("--- [5/5] Terminé ! Modèle sauvegardé. ---")

if __name__ == "__main__":
    run_training()