import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os  # Ajouté : nécessaire pour os.path.exists
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # Importation directe pour plus de clarté
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

# Algorithmes
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- 1. FONCTION D'IMPORTATION SÉCURISÉE ---
def charger_data(chemin_fichier):
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"❌ Le fichier '{chemin_fichier}' n'existe pas.")
    
    # Lecture flexible (détecte si c'est séparé par des virgules ou points-virgules)
    df = pd.read_csv(chemin_fichier, sep=None, engine='python')
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df

# --- 2. PRÉPARATION ET PIPELINE ---
try:
    # --- CHARGEMENT ---
    # Remplace par ton fichier réel (ex: 'data.csv')
    nom_fichier = 'data_transplantation.csv' 
    df = charger_data(nom_fichier) 
    
    # Séparation Features/Cible
    target = 'survie'
    if target not in df.columns:
        raise ValueError(f"❌ La colonne cible '{target}' est introuvable dans le fichier.")

    X = df.drop(target, axis=1)
    y = df[target]

    # --- ENCODAGE DE LA CIBLE (Indispensable pour XGBoost/LightGBM) ---
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"✅ Classes détectées : {list(le.classes_)} -> Transformées en : {np.unique(y)}")

    # Identification automatique des types de colonnes
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Split Train/Test (Stratifié pour préserver l'équilibre des classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- CONSTRUCTION DU PIPELINE DE PRÉTRAITEMENT ---
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # --- 3. DÉFINITION DES MODÈLES ---
    # Note : verbosity=-1 pour LightGBM évite les logs inutiles
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42)
    }

    # --- 4. ENTRAÎNEMENT ET COMPARAISON ---
    results = []
    best_auc = 0
    best_model = None
    best_model_name = ""

    print("\n🚀 Entraînement des modèles en cours...")

    for name, model in models.items():
        # Pipeline complet : Prétraitement + Algorithme
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Entraînement
        clf.fit(X_train, y_train)
        
        # Évaluation
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({"Modèle": name, "AUC": auc, "Accuracy": acc})
        print(f"📊 {name:15} | AUC: {auc:.4f} | Accuracy: {acc:.4f}")

        # Sauvegarde du champion
        if auc > best_auc:
            best_auc = auc
            best_model = clf
            best_model_name = name

    # --- 5. FINALISATION ---
    if best_model:
        joblib.dump(best_model, 'modele_final.pkl')
        # Sauvegarde aussi l'encodeur de labels pour le futur
        joblib.dump(le, 'label_encoder.pkl')
        print(f"\n🏆 Meilleur modèle sauvegardé : {best_model_name}")

    # --- 6. GRAPHIQUE DE PERFORMANCE ---
    plt.figure(figsize=(10, 6))
    df_res = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
    
    sns.set_theme(style="whitegrid")
    bar = sns.barplot(x='AUC', y='Modèle', data=df_res, hue='Modèle', palette='viridis', legend=False)
    
    plt.title('Comparaison des Modèles (HémoPredict)')
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")
    import traceback
    traceback.print_exc() # Affiche l'erreur précise pour le débogage