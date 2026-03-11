import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Chargement des données
from src.data_processing import load_and_clean_data
df = load_and_clean_data('data/bone-marrow.arff')

# 2. Séparation des caractéristiques (X) et de la cible (y)
# On suppose que la colonne à prédire s'appelle 'survie' (0 ou 1)
X = df.drop('survie', axis=1)
y = df['survie']

# 3. Identification des colonnes par type
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 4. Création des transformateurs (Nettoyage)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 5. Combinaison des transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Création du Pipeline final (Préparation + Modèle)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Division Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Entraînement
model_pipeline.fit(X_train, y_train)

# 9. Évaluation
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Sauvegarde du pipeline complet
joblib.dump(model_pipeline, 'modele_final.pkl')
print("Modèle sauvegardé avec succès !")
