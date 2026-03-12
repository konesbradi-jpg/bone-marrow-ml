import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --- 1. CHARGEMENT DES DONNÉES ---
try:
    # Remplace par le nom exact de ton fichier CSV
    df = pd.read_csv('data_transplantation.csv')
    print("Données chargées avec succès.")
except FileNotFoundError:
    print(" Erreur : Le fichier CSV est introuvable.")
    exit()

# Séparation des variables explicatives (X) et de la cible (y)
# On suppose que la colonne à prédire s'appelle 'survie' (0 = décès, 1 = survie)
X = df.drop('survie', axis=1)
y = df['survie']

# Séparation en jeu d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. PRÉTRAITEMENT ROBUSTE (PIPELINE) ---
# Identification des colonnes numériques et catégorielles
colonnes_numeriques = ['age', 'poids', 'hla_match']
colonnes_categorieles = ['type_donneur'] # Ajoute d'autres colonnes texte ici si besoin

# Création des transformateurs
transformateur_numerique = StandardScaler() # Met les valeurs sur la même échelle (Crucial pour le SVM)
transformateur_categoriel = OneHotEncoder(handle_unknown='ignore') # Transforme le texte en colonnes de 0 et 1

# Assemblage du prétraitement
preprocesseur = ColumnTransformer(
    transformers=[
        ('num', transformateur_numerique, colonnes_numeriques),
        ('cat', transformateur_categoriel, colonnes_categorieles)
    ])

# --- 3. DÉFINITION DES MODÈLES ---
# On configure les 3 modèles avec des paramètres de base solides
modeles = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # probability=True est OBLIGATOIRE pour le SVM, sinon la fonction predict_proba() de ton app Streamlit plantera
    "SVM": SVC(kernel='rbf', probability=True, random_state=42), 
    
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# --- 4. ENTRAÎNEMENT ET ÉVALUATION ---
print("\n🚀 Début de l'entraînement et de l'évaluation des modèles...\n")

resultats = {}
meilleur_modele = None
meilleur_score = 0
nom_meilleur_modele = ""

for nom, modele in modeles.items():
    # Création du pipeline complet : Prétraitement -> Modèle
    pipeline = Pipeline(steps=[('preprocessor', preprocesseur),
                               ('classifier', modele)])
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédictions sur le jeu de test
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # Probabilité de la classe 1 (survie)
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    resultats[nom] = {'Accuracy': acc, 'ROC-AUC': roc, 'Pipeline': pipeline}
    
    print(f"--- {nom} ---")
    print(f"Précision Globale (Accuracy) : {acc:.4f}")
    print(f"Score ROC-AUC              : {roc:.4f}")
    print("-" * 30)
    
    # On choisit le meilleur modèle basé sur le score ROC-AUC (plus pertinent pour les données médicales)
    if roc > meilleur_score:
        meilleur_score = roc
        meilleur_modele = pipeline
        nom_meilleur_modele = nom

# --- 5. SAUVEGARDE DU MEILLEUR MODÈLE ---
print(f"\n🏆 Le meilleur modèle est {nom_meilleur_modele} avec un score ROC-AUC de {meilleur_score:.4f}.")
print("💾 Sauvegarde en cours...")

# On sauvegarde le pipeline complet (qui inclut le modèle ET le nettoyeur de données)
joblib.dump(meilleur_modele, 'modele_final.pkl')

print("Modèle sauvegardé sous 'modele_final.pkl'. Il est prêt pour l'application Streamlit !")