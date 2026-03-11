import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement des données via votre module (Correction du format ARFF)
from src.data_processing import load_and_clean_data

print("Chargement des données en cours...")
df = load_and_clean_data('data/bone-marrow.arff')

# 2. Séparation des caractéristiques (X) et de la cible (y)
# Correction : Le nom identifié est 'survival_status'
target = 'survival_status'
X = df.drop(target, axis=1)
y = df[target]

# 3. Identification des colonnes par type
numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 4. Création des transformateurs (Nettoyage automatique)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # Correction : Utilisation de 'most_frequent' pour la cohérence médicale
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 5. Combinaison des transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Création du Pipeline final avec le modèle
# Correction : Ajout de class_weight='balanced' pour le déséquilibre (60/40)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    ))
])

# 7. Division Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Entraînement
print("Entraînement du modèle en cours...")
model_pipeline.fit(X_train, y_train)
from sklearn.svm import SVC

# --- CONFIGURATION DU 2ème MODÈLE (SVM) ---
model_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
])

# Entraînement du SVM
print("Entraînement du modèle SVM en cours...")
model_svm.fit(X_train, y_train)

# --- SAUVEGARDE DES DEUX MODÈLES ---
# On garde 'modele_final.pkl' pour la Random Forest (votre 1er modèle)
joblib.dump(model_pipeline, 'modele_rf.pkl') 

# On crée un nouveau fichier pour le SVM
joblib.dump(model_svm, 'modele_svm.pkl')

print("\nSuccès : Deux fichiers distincts ont été générés !")
print("- modele_rf.pkl (Random Forest)")
print("- modele_svm.pkl (SVM)")

# 9. Évaluation rapide
print("\n--- ÉVALUATION SUR LE SET DE TEST ---")
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Sauvegarde du modèle (indispensable pour le fichier d'évaluation)
joblib.dump(model_pipeline, 'modele_final.pkl')
print("\nSuccès : Le fichier 'modele_final.pkl' a été généré !")
from sklearn.ensemble import HistGradientBoostingClassifier

# --- CONFIGURATION DU 3ème MODÈLE (Gradient Boosting) ---
model_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
])

# Entraînement du Gradient Boosting
print("Entraînement du modèle Gradient Boosting en cours...")
model_gb.fit(X_train, y_train)

# Sauvegarde du troisième fichier
joblib.dump(model_gb, 'modele_gb.pkl')

print("\nFélicitations ! Le trio est complet :")
print("- modele_rf.pkl (Forêt Aléatoire)")
print("- modele_svm.pkl (SVM)")
print("- modele_gb.pkl (Gradient Boosting)")