import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.data_processing import load_and_clean_data

def run_evaluation(data_path, model_path):
    print("--- DÉMARRAGE DE L'ÉVALUATION ---")
    
    # 1. Chargement des données via votre fonction personnalisée
    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return

    # Correction du problème 'survie' : on prend la dernière colonne dynamiquement
    # Dans les fichiers ARFF, la cible est presque toujours en dernière position
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Colonnes utilisées pour prédire : {list(X.columns[:3])} ...")
    print(f"Variable cible identifiée : {y.name}")

    # 2. Chargement du modèle entraîné par votre camarade
    try:
        model = joblib.load(model_path)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur : Impossible de trouver le fichier {model_path}. Avez-vous lancé l'entraînement ?")
        return

    # 3. Prédictions
    y_pred = model.predict(X)
    # On récupère les probabilités pour la courbe ROC
    y_prob = model.predict_proba(X)[:, 1] 

    # 4. Affichage du rapport (Précision, Rappel pour le déséquilibre des classes)
    print("\n--- RÉSULTATS MÉDICAUX ---")
    print(classification_report(y, y_pred))

    # 5. Visualisation : Matrice de Confusion
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')

    # 6. Visualisation : Courbe ROC
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Performance ROC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # On ajoute 'data/' car c'est là que se trouve le fichier arff
    DATA_FILE = 'data/bone-marrow.arff' 
    MODEL_FILE = 'modele_final.pkl'
    run_evaluation(DATA_FILE, MODEL_FILE)